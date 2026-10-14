"""5-fold cross-validated training of the eye + head gate (4-way classification).

    python -m eyehead.train --folds_dir /content/drive/MyDrive/GazeVLM/folds \
                            --out_dir /content/drive/MyDrive/GazeVLM/runs/eyehead

Per fold, everything that is estimated from data -- the IMU->eye-axes map (in `fit` mode),
the channel normalisation and the class weights -- comes from that fold's TRAIN rows only.
Validation rows are never used for any of it.

Ablations use the same script:

    --streams eye             gaze only: what does the IMU add?
    --streams eye,head        no direction feature: what does cos(eye, head) add?
    --shuffle_control         features paired with another row's label; must fall to
                              chance (balanced accuracy ~0.25)

Reported CV numbers are the mean over the LAST 5 epochs, not the best epoch -- a maximum
over a noisy validation curve is biased upward even when there is no signal.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from gatetrain.data import check_thresholds, load_thresholds
from eyehead.features import (A_ASSUME_CPF, NAMES, TARGETS, class_profile,
                              compute_channels, fit_head_map, fit_stats, load_imu_calib,
                              load_rows, normalise, raw_signals, vor_check)
from eyehead.metrics import evaluate, report
from eyehead.model import EyeHeadGate, count_params

LAST_KEYS = ("acc", "bal_acc", "macro_f1", "macro_auc", "auc_gate", "recall_0", "recall_1",
             "recall_2", "recall_3", "gate_false_skip_rate", "gate_false_send_rate",
             "gate_send_rate")


def parse_args():
    ap = argparse.ArgumentParser("Eye + head 4-way gate, 5-fold CV")
    ap.add_argument("--folds_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thresholds", default=None,
                    help="thresholds.json; defaults to the folds dir's parent")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seq_len", type=int, default=9)

    ap.add_argument("--streams", default="eye,head,cos",
                    help="comma list from eye,head,cos -- the ablation knob")
    ap.add_argument("--imu_rot", default="auto", choices=["auto", "calib", "fit", "none"],
                    help="how to put the gyro in the eye's yaw/pitch axes; auto = calib "
                         "if --imu_calib is given, else fit (see features.py)")
    ap.add_argument("--imu_calib", default=None, help="JSON from eyehead.imu_calib")
    ap.add_argument("--eye_min", type=float, default=0.1,
                    help="rad/s; below this the eye direction is noise and cos = 0")
    ap.add_argument("--head_min", type=float, default=0.1,
                    help="rad/s; below this the head direction is noise and cos = 0")
    ap.add_argument("--fit_head_min", type=float, default=0.3)
    ap.add_argument("--fit_eye_max", type=float, default=1.0)
    ap.add_argument("--missing_imu", default="drop", choices=["drop", "zero"],
                    help="rows with no usable IMU: drop them from train/val, or keep "
                         "them with head = 0")

    ap.add_argument("--d", type=int, default=16, help="features per stream encoder")
    ap.add_argument("--h", type=int, default=32, help="fused features per step")
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--class_weight", default="inv", choices=["inv", "sqrt", "none"],
                    help="inv = N / (4 n_c): PURSUIT and REFIXATION count as much as "
                         "the two large classes")
    ap.add_argument("--select_on", default="bal_acc",
                    choices=["bal_acc", "macro_f1", "auc_gate", "min_recall"])
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--shuffle_control", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.streams = [s.strip() for s in a.streams.split(",") if s.strip()]
    return a


def uses_imu(streams):
    return "head" in streams or "cos" in streams


def resolve_head_map(a, E, G, M, ok):
    """The 2x3 map from raw gyro to (head_yaw, head_pitch) in the eye's convention."""
    if not uses_imu(a.streams):
        return A_ASSUME_CPF.copy(), dict(mode="unused")
    mode = a.imu_rot if a.imu_rot != "auto" else ("calib" if a.imu_calib else "fit")
    if mode == "calib":
        if not a.imu_calib:
            raise ValueError("--imu_rot calib needs --imu_calib")
        A, c = load_imu_calib(a.imu_calib)
        return A, dict(mode="calib", source=c.get("source_vrs"), A=A.tolist())
    if mode == "fit":
        A, info = fit_head_map(E, G, M, ok, a.fit_head_min, a.fit_eye_max)
        return A, dict(mode="fit", **info)
    return A_ASSUME_CPF.copy(), dict(mode="none", A=A_ASSUME_CPF.tolist())


def class_weights(y, kind):
    n = np.bincount(y, minlength=4).astype(np.float64)
    if kind == "none" or (n == 0).any():
        return np.ones(4, np.float32)
    w = len(y) / (4.0 * n)
    if kind == "sqrt":
        w = np.sqrt(w)
    return (w / w.mean()).astype(np.float32)


@torch.no_grad()
def predict_proba(model, Z, M, device, batch=4096):
    model.eval()
    out = []
    for s in range(0, len(Z), batch):
        x = torch.from_numpy(Z[s:s + batch]).to(device)
        m = torch.from_numpy(M[s:s + batch]).to(device)
        out.append(torch.softmax(model(x, m), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 4), np.float32)


def train_one_fold(csv_path, a, tau_f, tau_g, fold_id):
    torch.manual_seed(a.seed + fold_id); np.random.seed(a.seed + fold_id)
    df = load_rows(csv_path)
    check_thresholds(df[list(TARGETS)].to_numpy(np.float32), df["quad"].to_numpy(),
                     tau_f, tau_g, os.path.basename(csv_path))
    if "split" not in df.columns:
        raise KeyError(f"{csv_path} has no `split` column -- that is the test file")

    E, G, M, ok = raw_signals(df, a.seq_len)
    y = df["quad"].to_numpy().astype(int)
    keep = M.any(1)
    n_no_gaze = int((~keep).sum())
    # applied whatever the streams, so an eye-only ablation sees exactly the same rows
    if a.missing_imu == "drop":
        keep &= ok
    tr = keep & (df["split"] == "train").to_numpy()
    va = keep & (df["split"] == "val").to_numpy()
    print(f"   rows: train {tr.sum():,} / {df[tr].sequence.nunique()} videos   "
          f"val {va.sum():,} / {df[va].sequence.nunique()} videos   "
          f"(no gaze: {n_no_gaze} dropped;  no IMU: {int((M.any(1) & ~ok).sum())} "
          f"{'dropped' if a.missing_imu == 'drop' else 'kept with head = 0'})")

    # ---- everything estimated from data uses TRAIN rows only ------------------------
    A, hm = resolve_head_map(a, E[tr], G[tr], M[tr], ok[tr])
    X, H = compute_channels(E, G, M, A, a.eye_min, a.head_min)
    vor = vor_check(E[tr], H[tr], M[tr], ok[tr], a.fit_head_min, a.eye_min, a.fit_eye_max)
    stats = fit_stats(X[tr], M[tr])
    Z = normalise(X, M, stats)
    w = class_weights(y[tr], a.class_weight)

    if uses_imu(a.streams):
        extra = (f"   VOR gain {np.round(hm['vor_gain'], 2).tolist()}  fit R2 {hm['r2']:.3f}"
                 if hm["mode"] == "fit" else "")
        print(f"   IMU map: {hm['mode']}{extra}")
        verdict = ("ok" if vor["median_cos"] < -0.3 else
                   "!! NOT counter-rotating -- check the IMU map's signs/axes")
        circ = "  (circular in fit mode)" if hm["mode"] == "fit" else ""
        print(f"   VOR sign check: median cos {vor['median_cos']:+.3f} over {vor['n']:,} "
              f"head-turning smooth-eye steps, {100*vor['frac_negative']:.0f}% negative "
              f"-> {verdict}{circ}")
    print(f"   class weights {dict(zip(NAMES.values(), np.round(w, 2).tolist()))}")
    if fold_id == 1:
        print("\n   train-row profile (raw units; check it matches the physiology):")
        print("   " + class_profile(X[tr], M[tr], y[tr]).round(3).to_string()
              .replace("\n", "\n   ") + "\n")

    dev = a.device
    Xt, Mt, yt = (torch.from_numpy(Z[tr]).to(dev), torch.from_numpy(M[tr]).to(dev),
                  torch.from_numpy(y[tr]).to(dev))
    if a.shuffle_control:
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        perm = torch.randperm(len(yt), generator=g).to(dev)
        Xt, Mt = Xt[perm], Mt[perm]                 # features from a DIFFERENT row
        print("   *** SHUFFLE CONTROL: features paired with the wrong labels ***")
    Zv, Mv, yv = Z[va], M[va], y[va]

    model = EyeHeadGate(a.streams, a.d, a.h, a.kernel, dropout=a.dropout).to(dev)
    if fold_id == 1:
        print(f"   model: streams {a.streams}   {count_params(model):,} parameters")
    lossf = nn.CrossEntropyLoss(weight=torch.from_numpy(w).to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    gen = torch.Generator(device="cpu").manual_seed(a.seed + fold_id)

    best, best_state, hist, bad = -np.inf, None, [], 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(yt), generator=gen).to(dev)
        tot, n = 0.0, 0
        for s in range(0, len(perm), a.batch_size):
            j = perm[s:s + a.batch_size]
            loss = lossf(model(Xt[j], Mt[j]), yt[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(j); n += len(j)
        sched.step()

        pv = predict_proba(model, Zv, Mv, dev)
        m, _ = evaluate(pv, yv)
        m["val_loss"] = float(lossf(torch.log(torch.from_numpy(pv).clamp_min(1e-9)).to(dev),
                                    torch.from_numpy(yv).to(dev)))
        hist.append(dict(epoch=ep, tr_loss=tot / max(1, n), **m))

        score = m[a.select_on]
        improved = score > best + a.min_delta
        if improved:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep == 1 or ep % 10 == 0 or improved:
            print(f"   ep {ep:3d}  loss {tot/max(1,n):.4f}  |  val bal_acc {m['bal_acc']:.3f}"
                  f"  F1 {m['macro_f1']:.3f}  rec T/P/R/S {m['recall_0']:.2f}/"
                  f"{m['recall_1']:.2f}/{m['recall_2']:.2f}/{m['recall_3']:.2f}"
                  f"  AUC_gate {m['auc_gate']:.3f}{'  *' if improved else ''}")
        if bad >= a.patience:
            print(f"   early stop at epoch {ep} (no {a.select_on} gain for {a.patience})")
            break

    k = min(5, len(hist))
    last = {kk: float(np.nanmean([h[kk] for h in hist[-k:]])) for kk in LAST_KEYS}
    last["epochs_averaged"] = k
    best_ep = int(np.argmax([h[a.select_on] for h in hist])) + 1
    print(f"   best {a.select_on} {best:.4f} @ep{best_ep}  |  last-{k} bal_acc "
          f"{last['bal_acc']:.4f}  REFIX recall {last['recall_2']:.3f}   ({time.time()-t0:.0f}s)")

    model.load_state_dict(best_state)
    mb, _ = evaluate(predict_proba(model, Zv, Mv, dev), yv)
    print(report(mb, f"   val @ best epoch (fold {fold_id}):"))

    ck = dict(model=best_state, fold=fold_id, best=float(best), best_epoch=best_ep, last=last,
              args=vars(a), streams=a.streams,
              arch=dict(d=a.d, h=a.h, k=a.kernel, dropout=a.dropout),
              head_map=np.asarray(A).tolist(), head_map_info=hm, vor_check=vor,
              stats=[stats[0].tolist(), stats[1].tolist()], class_weights=w.tolist(),
              eye_min=a.eye_min, head_min=a.head_min, seq_len=a.seq_len,
              tau_frame=tau_f, tau_gaze=tau_g, n_params=count_params(model))
    return ck, hist


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    th = a.thresholds or os.path.join(os.path.dirname(a.folds_dir.rstrip("/")),
                                      "thresholds.json")
    tau_f, tau_g, _ = load_thresholds(th)
    print(f"streams    : {a.streams}")
    print(f"IMU map    : {a.imu_rot}{f'  ({a.imu_calib})' if a.imu_calib else ''}")
    print(f"thresholds : tau_frame {tau_f:.4f}  tau_gaze {tau_g:.4f}  (label check only)")
    print(f"loss       : cross-entropy, class weights '{a.class_weight}'")
    print(f"device     : {a.device}")
    if a.shuffle_control:
        print("\n*** SHUFFLE CONTROL RUN -- balanced accuracy must stay near 0.25 ***")

    results, all_hist = [], {}
    for k in range(1, a.k + 1):
        csv = os.path.join(a.folds_dir, f"train_val_fold{k}.csv")
        if not os.path.exists(csv):
            raise FileNotFoundError(csv)
        print(f"\n{'='*78}\nFOLD {k}/{a.k}\n{'='*78}")
        ck, hist = train_one_fold(csv, a, tau_f, tau_g, k)
        path = os.path.join(a.out_dir, f"fold{k}.pt")
        torch.save(ck, path)
        print(f"   saved -> {path}")
        results.append({kk: ck[kk] for kk in ("fold", "best", "best_epoch", "last",
                                              "head_map_info", "vor_check")})
        all_hist[f"fold{k}"] = hist

    L = lambda key: np.array([r["last"][key] for r in results], float)
    print(f"\n{'='*92}\nCROSS-VALIDATION SUMMARY  ({a.k} folds, mean of last 5 epochs)\n{'='*92}")
    cols = ("bal_acc", "macro_f1", "auc_gate", "recall_0", "recall_1", "recall_2",
            "recall_3", "gate_false_skip_rate")
    head = ("bal_acc", "macro_F1", "AUC_gate", "rec_TRAN", "rec_PURS", "rec_REFX",
            "rec_STAB", "false_skip")
    print(f"{'fold':>5} " + " ".join(f"{h:>9}" for h in head))
    for r in results:
        print(f"{r['fold']:>5} " + " ".join(f"{r['last'][c]:>9.3f}" for c in cols))
    print("-" * 92)
    print(f"{'mean':>5} " + " ".join(f"{L(c).mean():>9.3f}" for c in cols))
    print(f"{'std':>5} " + " ".join(f"{L(c).std():>9.3f}" for c in cols))
    print("=" * 92)
    print("Chance balanced accuracy is 0.25. The previous gaze-only gate's test recall was")
    print("TRANSITION 0.93 / PURSUIT 0.09 / REFIXATION 0.00 / STABLE 0.60.")

    json.dump(dict(args=vars(a), tau_frame=tau_f, tau_gaze=tau_g, folds=results,
                   history=all_hist, **{f"{c}_mean": float(L(c).mean()) for c in cols},
                   **{f"{c}_std": float(L(c).std()) for c in cols}, headline="last5"),
              open(os.path.join(a.out_dir, "cv_results.json"), "w"), indent=2, default=str)
    print(f"\nwrote {a.out_dir}/cv_results.json")


if __name__ == "__main__":
    main()
