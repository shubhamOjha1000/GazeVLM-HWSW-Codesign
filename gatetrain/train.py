"""5-fold cross-validated training of the PDF pipeline (p2 + p3, summed per p9).

    python -m gatetrain.train --folds_dir /content/drive/MyDrive/GazeVLM/folds \
                              --out_dir runs/joint

What is trained: f_M turns the gaze rates into z_M; a head regresses [S_frame, S_gaze]
from z_M (L_sim); and a change predictor h reconstructs z_M from the two frozen frame
features so InfoNCE can pull them together (L_NCE). Only f_M and the head are deployed --
h exists to create the Loss-1 target and is discarded at inference.

The reported figure is the mean over the LAST few epochs, not the best epoch. A maximum
over a noisy validation curve is a multiple-comparisons trap: measured on this project, a
shuffle control with no signal by construction still reached AUC 0.90 on its best epoch.
The checkpoint keeps the best epoch, which is legitimate model selection, and the test set
stays independent of that choice.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from gatetrain.data import (FeatureCache, FoldSet, check_thresholds, load_thresholds,
                            train_stats)
from gatetrain.losses import evaluate, joint_loss
from gatetrain.model import build, count_params

QUAD = {0: "TRANSITION", 1: "PURSUIT", 2: "REFIXATION", 3: "STABLE"}


def parse_args():
    ap = argparse.ArgumentParser("Joint L_NCE + L_sim training over the fold CSVs")
    ap.add_argument("--folds_dir", required=True)
    ap.add_argument("--out_dir", default="runs/joint")
    ap.add_argument("--thresholds", default=None,
                    help="thresholds.json from the labelling notebook; defaults to "
                         "<folds_dir>/../thresholds.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--feat_root", default=None,
                    help="re-root the .npz paths if the features moved")
    ap.add_argument("--feat_cache", default="/content/feat_cache.npz",
                    help="local copy of the frame-feature matrix, so Drive is read once")
    ap.add_argument("--sim_only", action="store_true",
                    help="train L_sim alone (PDF p3). Skips the .npz entirely -- useful "
                         "when the features are unavailable, and a clean ablation of "
                         "what the contrastive term is worth")

    ap.add_argument("--rates_col", default="gaze_rates_window",
                    choices=["gaze_rates_window", "gaze_vec3d_rates_window"])
    ap.add_argument("--seq_len", type=int, default=9)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--tau", type=float, default=0.1)          # PDF p2
    ap.add_argument("--nce_weight", type=float, default=1.0)   # PDF p9: plain sum
    ap.add_argument("--max_per_video", type=int, default=8,
                    help="cap rows of one video per batch; InfoNCE negatives come from "
                         "the batch and two windows seconds apart are false negatives")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")

    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--select_on", default="auc_gate",
                    choices=["auc_gate", "r2_mean", "auc_frame"],
                    help="validation metric for checkpointing and early stopping")

    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--fusion", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shuffle_control", action="store_true",
                    help="CONTROL: pair each row's frames with ANOTHER row's gaze. Must "
                         "fail; if it does not, the result is an artefact of the setup.")
    return ap.parse_args()


def make_scaler(enabled):
    """GradScaler across torch versions.

    torch.amp.GradScaler("cuda", ...) is the 2.x spelling; torch.cuda.amp.GradScaler is
    the older one and is deprecated in 2.x. Colab has 2.11, this repo also runs on 1.x,
    so try the new form and fall back rather than pinning either.
    """
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def predict_raw(model, data, feats, batch_size, tmean, tstd, amp=False):
    """Predictions in RAW cosine units -- the loss trains on standardised targets, but the
    thresholds live in raw units, so un-standardising here is not optional."""
    model.eval()
    out = []
    for j in data.batches(batch_size):
        X, M, _, _, _ = data.gather(j, feats)
        with torch.autocast("cuda", enabled=amp and X.is_cuda):
            out.append(model.head(model.f_m(X, M)).float().cpu().numpy())
    p = np.concatenate(out) if out else np.zeros((0, 2), np.float32)
    return p * tstd + tmean


def run_epoch(model, data, a, feats, opt=None, scaler=None, gen=None, seq_ids=None):
    train = opt is not None
    model.train(train)
    agg, n = {}, 0
    for j in data.batches(a.batch_size, shuffle=train, generator=gen,
                          max_per_video=a.max_per_video if train else None,
                          seq_ids=seq_ids):
        X, M, y, z0, zT = data.gather(j, feats)
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", enabled=a.amp and X.is_cuda):
                pred, z_m, z_hat = model(X, M, z0, zT)
                loss, parts = joint_loss(pred, y, z_hat, z_m, a.tau, a.nce_weight)
        if train:
            opt.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        b = len(j)
        for k, v in parts.items():
            if np.isfinite(v):
                agg[k] = agg.get(k, 0.0) + v * b
        n += b
    return {k: v / max(1, n) for k, v in agg.items()}


def train_one_fold(csv_path, a, cache, tau_f, tau_g, fold_id):
    torch.manual_seed(a.seed + fold_id); np.random.seed(a.seed + fold_id)

    # Statistics from the TRAIN rows only, read straight from the CSV. Do NOT take these
    # off a subset of a whole-file FoldSet: subset() inherits the parent's stats, so that
    # route silently normalises with val rows included.
    rate_stats, target_stats = train_stats(csv_path, a.rates_col, a.seq_len)
    full = FoldSet(csv_path, cache, a.rates_col, a.seq_len, a.device,
                   rate_stats=rate_stats, target_stats=target_stats,
                   feat_root=a.feat_root)
    check_thresholds(full.y_raw_np, full.quad_np, tau_f, tau_g,
                     os.path.basename(csv_path))
    tr, va = full.train_val()
    tmean, tstd = target_stats
    print(f"   train {len(tr):,} rows / {tr.df.sequence.nunique()} videos"
          f"   val {len(va):,} rows / {va.df.sequence.nunique()} videos")

    if a.shuffle_control:
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        perm = torch.randperm(len(tr), generator=g).to(a.device)
        tr.X, tr.M = tr.X[perm], tr.M[perm]      # gaze from a DIFFERENT row than the frames
        print("   *** SHUFFLE CONTROL: gaze paired with the wrong frames ***")

    feats = None
    if not a.sim_only:
        feats = torch.from_numpy(cache.F).to(a.device)

    model = build(feat_dim=(cache.dim if cache else 384), dim=a.dim, hidden=a.hidden,
                  width=a.width, fusion=a.fusion, dropout=a.dropout).to(a.device)
    if fold_id == 1:
        c = count_params(model)
        print(f"   params: total {c['total']:,}   f_M {c['f_m']:,}   h {c['h']:,}   "
              f"head {c['head']:,}   DEPLOYED (f_M+head) {c['deployed']:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs,
                                                pct_start=0.1)
    scaler = make_scaler(a.amp and a.device == "cuda")
    gen = torch.Generator(device="cpu").manual_seed(a.seed + fold_id)
    seq_ids = tr.video_ids()

    best, best_state, hist, bad = -np.inf, None, [], 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        trp = run_epoch(model, tr, a, feats, opt, scaler, gen, seq_ids)
        pv = predict_raw(model, va, feats, a.batch_size, tmean, tstd, a.amp)
        m, _ = evaluate(pv, va.y_raw_np, va.quad_np, tau_f, tau_g)
        sched.step()
        hist.append(dict(epoch=ep, **{f"tr_{k}": v for k, v in trp.items()}, **m))

        score = m[a.select_on]
        improved = score > best + a.min_delta
        if improved:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep == 1 or ep % 10 == 0 or improved:
            nce = f"  nce {trp.get('nce', float('nan')):.3f}" if not a.sim_only else ""
            print(f"   ep {ep:3d}/{a.epochs}  sim {trp['sim']:.4f}{nce}  |  "
                  f"val AUC_gate {m['auc_gate']:.4f}  R2 {m['r2_mean']:+.3f}  "
                  f"quad_acc {m['quad_acc']:.3f}{'  *' if improved else ''}")
        if bad >= a.patience:
            print(f"   early stop at epoch {ep} (no {a.select_on} gain for {a.patience})")
            break

    k = min(5, len(hist))
    last = {kk: float(np.mean([h[kk] for h in hist[-k:]]))
            for kk in ("auc_gate", "auc_frame", "auc_gaze", "r2_mean", "quad_acc",
                       "recall_q2", "gate_false_skip_rate", "gate_false_send_rate")}
    last["epochs_averaged"] = k
    print(f"   best {a.select_on} {best:.4f} @ep{int(np.argmax([h[a.select_on] for h in hist]))+1}"
          f"  |  last-{k} AUC_gate {last['auc_gate']:.4f}   ({time.time()-t0:.0f}s)")
    return best_state, hist, dict(best=float(best), last=last), (rate_stats, target_stats)


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    th_path = a.thresholds or os.path.join(os.path.dirname(a.folds_dir.rstrip("/")),
                                           "thresholds.json")
    tau_f, tau_g, meta = load_thresholds(th_path)

    print(f"objective   : L_sim{'' if a.sim_only else f' + {a.nce_weight} * L_NCE'}"
          f"   (PDF p3{'' if a.sim_only else ' + p2, summed per p9'})")
    print(f"thresholds  : tau_frame {tau_f:.4f}  tau_gaze {tau_g:.4f}   from {th_path}")
    print(f"              (percentiles {meta.get('frame_pct')}/{meta.get('gaze_pct')}, "
          f"per_video={meta.get('per_video')})")
    print(f"input signal: {a.rates_col}")
    print(f"device      : {a.device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if a.device == "cuda" else ""))
    if a.shuffle_control:
        print("\n*** SHUFFLE CONTROL RUN -- R2 must stay <= 0 and AUC near 0.5 ***")

    cache = None
    if not a.sim_only:
        import pandas as pd
        f1 = os.path.join(a.folds_dir, "train_val_fold1.csv")
        cols = pd.read_csv(f1, usecols=["feat_frame_1", "feat_frame_2"])
        te = os.path.join(a.folds_dir, "test.csv")
        allp = list(cols.feat_frame_1) + list(cols.feat_frame_2)
        if os.path.exists(te):
            t = pd.read_csv(te, usecols=["feat_frame_1", "feat_frame_2"])
            allp += list(t.feat_frame_1) + list(t.feat_frame_2)
        print(f"\nbuilding the frame-feature cache (read once, reused by every fold)")
        cache = FeatureCache(allp, a.feat_root, cache_file=a.feat_cache)

    results, all_hist = [], {}
    for k in range(1, a.k + 1):
        csv = os.path.join(a.folds_dir, f"train_val_fold{k}.csv")
        if not os.path.exists(csv):
            raise FileNotFoundError(csv)
        print(f"\n{'='*78}\nFOLD {k}/{a.k}\n{'='*78}")
        state, hist, r, stats = train_one_fold(csv, a, cache, tau_f, tau_g, k)
        ck = os.path.join(a.out_dir, f"fold{k}.pt")
        torch.save(dict(model=state, args=vars(a), fold=k, **r,
                        tau_frame=tau_f, tau_gaze=tau_g,
                        rate_stats=[stats[0][0].tolist(), stats[0][1].tolist()],
                        target_stats=[stats[1][0].tolist(), stats[1][1].tolist()],
                        feat_dim=(cache.dim if cache else 384)), ck)
        results.append(r); all_hist[f"fold{k}"] = hist
        print(f"   saved -> {ck}")

    L = lambda key: np.array([r["last"][key] for r in results], float)
    auc, r2, q2 = L("auc_gate"), L("r2_mean"), L("recall_q2")

    print(f"\n{'='*84}\nCROSS-VALIDATION SUMMARY  ({a.k} folds)\n{'='*84}")
    print(f"{'fold':>5} {'AUC_gate':>10} {'AUC_frame':>10} {'AUC_gaze':>9} "
          f"{'R2':>8} {'quad_acc':>9} {'REFIX_rec':>10}")
    for i, r in enumerate(results, 1):
        d = r["last"]
        print(f"{i:>5} {d['auc_gate']:>10.4f} {d['auc_frame']:>10.4f} "
              f"{d['auc_gaze']:>9.4f} {d['r2_mean']:>8.3f} {d['quad_acc']:>9.3f} "
              f"{d['recall_q2']:>10.3f}")
    print("-" * 84)
    print(f"{'mean':>5} {auc.mean():>10.4f} {L('auc_frame').mean():>10.4f} "
          f"{L('auc_gaze').mean():>9.4f} {r2.mean():>8.3f} "
          f"{L('quad_acc').mean():>9.3f} {q2.mean():>10.3f}")
    print(f"{'std':>5} {auc.std():>10.4f} {L('auc_frame').std():>10.4f} "
          f"{L('auc_gaze').std():>9.4f} {r2.std():>8.3f} "
          f"{L('quad_acc').std():>9.3f} {q2.std():>10.3f}")
    print("=" * 84)
    print("All figures are the mean over the last 5 epochs, not the best epoch: a maximum")
    print("over a noisy validation curve is biased upward even when no signal exists.")

    if r2.mean() <= 0:
        print("\n  R2 <= 0: the head is no better than predicting the training mean on")
        print("  held-out videos. AUC can still look respectable when that is true, so")
        print("  read R2 first -- it has a baseline, AUC does not.")
    if np.isfinite(q2).all() and q2.mean() < 0.1:
        print("\n  REFIXATION recall is near zero. That quadrant is the entire reason the")
        print("  design has two thresholds instead of one, so this is the result to fix")
        print("  before claiming the two-score system earns its keep.")

    json.dump(dict(args=vars(a), tau_frame=tau_f, tau_gaze=tau_g,
                   folds=results, history=all_hist,
                   auc_gate_mean=float(auc.mean()), auc_gate_std=float(auc.std()),
                   r2_mean=float(r2.mean()), headline="last5"),
              open(os.path.join(a.out_dir, "cv_results.json"), "w"), indent=2, default=str)
    print(f"\nwrote {a.out_dir}/cv_results.json")


if __name__ == "__main__":
    main()
