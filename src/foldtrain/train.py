"""Cross-validated training over the 5 fold CSVs, with early stopping.

    python -m src.foldtrain.train --folds_dir /content/drive/MyDrive/GazeVLM/folds \
                                  --out_dir runs/gate --target gate

Trains one model per fold, keeps the best epoch by validation AUC, and reports the mean
and spread across folds. The spread is the point of doing this: a single split of ~12
videos gives one noisy number, and five give an estimate of how much of it was the luck of
which videos landed in validation.

On T4 settings
--------------
The dominant cost in this job is NOT the GPU. The model is ~300k parameters over a 9-step
sequence, and the whole dataset is ~1.4 MB of floats. Once `FoldData` puts it on the
device (see dataset.py) there is no host-to-device traffic per step at all, and an epoch
is a few hundred kernel launches.

So the settings that matter are:

  batch_size 512   large, because the limit is launch overhead rather than memory;
                   raising it further mostly stops helping once an epoch is <10 batches
  amp        on    T4 has tensor cores. The gain is modest at this size -- the model is
                   too small to be matmul-bound -- but it is free and does not hurt
  workers    none  there is no DataLoader to have workers

What would NOT help: more workers, pin_memory, prefetching, a bigger GPU. If you want this
faster, the lever is fewer epochs via early stopping, which is why patience defaults low.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.foldtrain.dataset import FoldData, GATE_NAMES, QUAD_NAMES
from src.foldtrain.metrics import summarise
from src.foldtrain.models import build, count_params, load_encoder_from


def parse_args():
    ap = argparse.ArgumentParser("Cross-validated gate classifier")
    ap.add_argument("--folds_dir", required=True,
                    help="directory holding train_val_fold1..K.csv and test.csv")
    ap.add_argument("--out_dir", default="runs/gate")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--target", default="gate", choices=["gate", "quad"],
                    help="'gate' = binary SEND/DISCARD, 'quad' = the 4 quadrants")
    ap.add_argument("--rates_col", default="gaze_rates_window",
                    choices=["gaze_rates_window", "gaze_vec3d_rates_window"],
                    help="gaze_vec3d_rates_window is sphere-exact; omega_mag over-reads "
                         "by 1/cos(pitch)")
    ap.add_argument("--seq_len", type=int, default=9)

    # --- optimisation (tuned for a Colab T4; see the module docstring) ---
    ap.add_argument("--epochs", type=int, default=200, help="cap; early stopping usually hits first")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")

    # --- early stopping ---
    ap.add_argument("--patience", type=int, default=20,
                    help="epochs without val-AUC improvement before stopping")
    ap.add_argument("--min_delta", type=float, default=1e-4,
                    help="improvement below this does not count as improvement")

    # --- capacity ---
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--class_weights", action="store_true", default=True,
                    help="inverse-frequency weighting; on by default because REFIXATION "
                         "is rare and is the class the design exists for")
    ap.add_argument("--no_class_weights", dest="class_weights", action="store_false")
    ap.add_argument("--init_from", default=None,
                    help="warm-start f_M from a Loss-1/Loss-2 checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shuffle_control", action="store_true",
                    help="CONTROL: shuffle the labels. Must land at AUC ~0.5; if it does "
                         "not, the evaluation is leaking and no real result is credible.")
    return ap.parse_args()


@torch.no_grad()
def predict(model, data, batch_size, amp=False):
    """Class probabilities for every row, in order."""
    model.eval()
    out = []
    for X, M, _ in data.batches(batch_size):
        with torch.autocast("cuda", enabled=amp and X.is_cuda):
            out.append(torch.softmax(model(X, M).float(), dim=-1))
    return torch.cat(out).cpu().numpy() if out else np.zeros((0, model.n_classes))


def run_epoch(model, data, crit, batch_size, opt=None, scaler=None, amp=False, gen=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    for X, M, y in data.batches(batch_size, shuffle=train, generator=gen):
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", enabled=amp and X.is_cuda):
                loss = crit(model(X, M), y)
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
        tot += float(loss.detach()) * len(y); n += len(y)
    return tot / max(1, n)


def train_one_fold(csv_path, a, fold_id):
    """Returns (best_state, history, best_metrics, stats)."""
    torch.manual_seed(a.seed + fold_id); np.random.seed(a.seed + fold_id)

    full = FoldData(csv_path, target=a.target, rates_col=a.rates_col,
                    seq_len=a.seq_len, device=a.device)
    tr, va = full.train_val()
    # val is standardised with the TRAIN statistics of its own fold -- recomputing them on
    # val would be a (small) leak and would make folds incomparable
    tr.describe("train"); va.describe("val")

    if a.shuffle_control:
        g = torch.Generator(device=tr.y.device).manual_seed(a.seed)
        tr.y = tr.y[torch.randperm(len(tr), device=tr.y.device, generator=g)]
        tr.y_np = tr.y.cpu().numpy()
        print("   *** SHUFFLE CONTROL: training labels permuted ***")

    model = build(n_classes=full.n_classes, width=a.width, dim=a.dim,
                  hidden=a.hidden, dropout=a.dropout).to(a.device)
    if a.init_from:
        load_encoder_from(model, a.init_from, a.device)
        print(f"   warm-started f_M from {a.init_from}")

    w = tr.class_weights() if a.class_weights else None
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=a.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs,
        pct_start=min(0.3, a.warmup / max(1, a.epochs)))
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and a.device == "cuda")
    gen = torch.Generator(device=tr.X.device).manual_seed(a.seed + fold_id)

    best = dict(auc=-np.inf, epoch=-1)
    best_state, hist, bad = None, [], 0
    t0 = time.time()

    for ep in range(1, a.epochs + 1):
        trl = run_epoch(model, tr, crit, a.batch_size, opt, scaler, a.amp, gen)
        val_p = predict(model, va, a.batch_size, a.amp)
        m = summarise(va.y_np, val_p, a.target)
        vll = float(crit(torch.log(torch.tensor(val_p).clamp_min(1e-9)).to(a.device),
                         va.y))          # for the curve only
        sched.step()
        hist.append(dict(epoch=ep, train_loss=trl, val_loss=vll, **m))

        improved = m["auc"] > best["auc"] + a.min_delta
        if improved:
            best = dict(**m, epoch=ep)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep % 10 == 0 or improved or ep == 1:
            print(f"   ep {ep:3d}/{a.epochs}  loss {trl:.4f}  val AUC {m['auc']:.4f}  "
                  f"acc {m['acc']:.3f}  bal {m['bal_acc']:.3f}"
                  f"{'   *' if improved else ''}")
        if bad >= a.patience:
            print(f"   early stop at epoch {ep}: no AUC gain for {a.patience} epochs")
            break

    # Three numbers, because the obvious one is biased.
    #
    # `best` is the max of a noisy validation AUC over every epoch. On a genuinely
    # signal-free run that max still lands high: a shuffle control on this data reached
    # 0.909 at epoch 4 while its balanced accuracy sat at exactly chance, i.e. while it
    # was predicting a single class. Reporting a max over epochs is a multiple-comparisons
    # trap.
    #
    # So the checkpoint keeps the best epoch -- that is legitimate model selection, and
    # test stays independent of it -- but the REPORTED figure is the mean over the last
    # few epochs, which no amount of epoch-shopping can inflate.
    k = min(5, len(hist))
    last = dict(
        auc=float(np.mean([h["auc"] for h in hist[-k:]])),
        acc=float(np.mean([h["acc"] for h in hist[-k:]])),
        bal_acc=float(np.mean([h["bal_acc"] for h in hist[-k:]])),
        epochs_averaged=k,
    )
    final = hist[-1]
    print(f"   best epoch {best['epoch']}  AUC {best['auc']:.4f}  |  "
          f"final {final['auc']:.4f}  |  last-{k} mean {last['auc']:.4f}   "
          f"({time.time()-t0:.0f}s, {len(hist)} epochs)")

    chance = 1.0 / full.n_classes
    if last["bal_acc"] < chance + 0.02:
        print(f"   !! balanced accuracy {last['bal_acc']:.3f} is at chance ({chance:.2f}):")
        print(f"      the model is predicting essentially one class. Any AUC above 0.5")
        print(f"      here is ranking noise, not a usable classifier.")
    return best_state, hist, dict(best=best, final=final, last=last), full.stats


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    names = GATE_NAMES if a.target == "gate" else QUAD_NAMES

    print(f"target      : {a.target}  ({len(names)} classes: {list(names.values())})")
    print(f"input signal: {a.rates_col}")
    print(f"device      : {a.device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if a.device == "cuda" else ""))
    print(f"amp         : {a.amp}   batch {a.batch_size}   lr {a.lr}   "
          f"max epochs {a.epochs}   patience {a.patience}")
    if a.shuffle_control:
        print("\n*** SHUFFLE CONTROL RUN -- expect AUC ~0.5 on every fold ***")

    results, all_hist = [], {}
    for k in range(1, a.k + 1):
        csv = os.path.join(a.folds_dir, f"train_val_fold{k}.csv")
        if not os.path.exists(csv):
            raise FileNotFoundError(csv)
        print(f"\n{'='*74}\nFOLD {k}/{a.k}   {os.path.basename(csv)}\n{'='*74}")
        state, hist, r, stats = train_one_fold(csv, a, k)

        ck = os.path.join(a.out_dir, f"fold{k}.pt")
        torch.save(dict(model=state, args=vars(a), fold=k, best=r["best"],
                        final=r["final"], last=r["last"],
                        stats=[stats[0].tolist(), stats[1].tolist()],
                        n_classes=len(names)), ck)
        results.append(r); all_hist[f"fold{k}"] = hist
        print(f"   saved -> {ck}")

    # headline = mean over the last epochs, NOT the per-fold maximum (see train_one_fold)
    aucs = np.array([r["last"]["auc"] for r in results], dtype=float)
    bals = np.array([r["last"]["bal_acc"] for r in results], dtype=float)
    bests = np.array([r["best"]["auc"] for r in results], dtype=float)

    print(f"\n{'='*80}\nCROSS-VALIDATION SUMMARY  ({a.k} folds, target={a.target})\n{'='*80}")
    print(f"{'fold':>5} {'AUC(last5)':>11} {'bal_acc':>9} {'AUC(best_ep)':>13} {'best_ep':>8}")
    for i, r in enumerate(results, 1):
        print(f"{i:>5} {r['last']['auc']:>11.4f} {r['last']['bal_acc']:>9.4f} "
              f"{r['best']['auc']:>13.4f} {r['best']['epoch']:>8}")
    print("-" * 80)
    print(f"{'mean':>5} {aucs.mean():>11.4f} {bals.mean():>9.4f} {bests.mean():>13.4f}")
    print(f"{'std':>5} {aucs.std():>11.4f} {bals.std():>9.4f} {bests.std():>13.4f}")
    print(f"{'range':>5} {aucs.min():>11.4f} .. {aucs.max():.4f}")
    print("=" * 80)
    print("REPORT THE AUC(last5) COLUMN. AUC(best_ep) is a maximum over every epoch of a")
    print("noisy validation curve and is biased upward -- on a shuffled control it still")
    print(f"reaches high values. Here the gap is {bests.mean()-aucs.mean():+.4f}.")

    chance = 1.0 / len(names)
    if bals.mean() < chance + 0.02:
        print("\n  BALANCED ACCURACY IS AT CHANCE. The model predicts one class; the AUC")
        print("  above is ranking noise. Treat this run as a failure regardless of AUC.")
    if aucs.std() > 0.05:
        print("\n  The fold spread is wide. Most of the headline number is then the luck")
        print("  of which videos landed in validation, not the model. Report the spread.")
    if aucs.mean() < 0.55:
        print("\n  AUC near 0.5 means gaze is not separating these classes at this scale.")
        print("  Check the shuffle control lands at the same place -- if it does not, the")
        print("  evaluation is leaking rather than the signal being absent.")

    json.dump(dict(args=vars(a), folds=results, history=all_hist,
                   auc_mean=float(aucs.mean()), auc_std=float(aucs.std()),
                   auc_best_mean=float(bests.mean()),
                   headline="last5"),
              open(os.path.join(a.out_dir, "cv_results.json"), "w"), indent=2, default=str)
    print(f"\nwrote {a.out_dir}/cv_results.json")


if __name__ == "__main__":
    main()
