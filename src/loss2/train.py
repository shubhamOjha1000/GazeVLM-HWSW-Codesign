"""Loss-2 training: predict [S_frame, S_gaze] from the gaze-rate signal alone.

    python -m src.loss2.train --csv data/feature_dataset.csv --out_dir runs/loss2
    python -m src.loss2.train --csv ... --joint                # L = L_sim + L_NCE
    python -m src.loss2.train --csv ... --shuffle_control      # negative control

Why this is the more informative experiment
-------------------------------------------
Loss 2 is a regression with a trivial baseline: predict the training mean for every row.
A model that cannot beat that on HELD-OUT VIDEOS has found no signal -- no ambiguity
about memorisation, no best-epoch cherry-picking, no dependence on batch composition.
Loss 1's contrastive retrieval offers no such reference point.

It is also the more direct test of the premise: the gate needs S_frame and S_gaze
predicted from gaze alone, and that is exactly this task. Loss 1 only shapes the
representation that feeds it.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.loss1.dataset import (VideoBalancedBatchSampler, compute_channel_stats,
                               split_by_sequence)
from src.loss1.losses import info_nce, retrieval_metrics
from src.loss1.models import ChangePredictor, GazeRateEncoder
from src.loss2.dataset import Loss2Dataset, compute_target_stats
from src.loss2.losses import TARGETS, format_metrics, regression_metrics, similarity_loss
from src.loss2.models import build_head


def parse_args():
    ap = argparse.ArgumentParser("Loss-2 similarity regression")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="runs/loss2")
    ap.add_argument("--feat_root", default=None)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--width", type=int, default=128, help="gaze-encoder conv width")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seq_len", type=int, default=9)

    ap.add_argument("--joint", action="store_true",
                    help="also train Loss 1: L = L_sim + nce_weight * L_NCE")
    ap.add_argument("--nce_weight", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--init_from", default=None,
                    help="path to a Loss-1 best.pt to warm-start the gaze encoder")

    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--max_per_video", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--shuffle_control", action="store_true",
                    help="CONTROL: pair each frame pair with ANOTHER row's gaze rates. "
                         "Must fail to beat predict-the-mean.")
    return ap.parse_args()


def run_epoch(f_m, head, h, loader, a, tmean, tstd, opt=None):
    """Returns (loss, metrics). Predictions are gathered across the whole split so R2
    is computed once over all rows rather than averaged over batches."""
    train = opt is not None
    for m in (f_m, head):
        m.train(train)
    if h is not None:
        h.train(train)

    tot, n, P, Y, nce_top1, nce_n = 0.0, 0, [], [], 0.0, 0
    for b in loader:
        rates = b["rates"].to(a.device, non_blocking=True)
        mask = b["mask"].to(a.device, non_blocking=True)
        y = b["y"].to(a.device, non_blocking=True)

        with torch.set_grad_enabled(train):
            z_m = f_m(rates, mask)
            pred = head(z_m)
            loss, _ = similarity_loss(pred, y)

            if a.joint and h is not None and rates.size(0) >= 2:
                z_hat = h(b["z0"].to(a.device), b["zt"].to(a.device))
                nce, logits = info_nce(z_hat, z_m, a.tau)
                loss = loss + a.nce_weight * nce
                mm = retrieval_metrics(logits.detach())
                nce_top1 += mm["top1"] * rates.size(0); nce_n += rates.size(0)

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            params = list(f_m.parameters()) + list(head.parameters())
            if h is not None:
                params += list(h.parameters())
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

        bs = rates.size(0)
        tot += loss.item() * bs; n += bs
        P.append(pred.detach().cpu().numpy())
        Y.append(b["y_raw"].numpy())

    if n == 0:
        return float("nan"), None
    # back to raw similarity units, so R2 and r mean something
    pred_raw = np.concatenate(P) * tstd + tmean
    met = regression_metrics(pred_raw, np.concatenate(Y), train_mean=tmean)
    if nce_n:
        met["nce_top1"] = nce_top1 / nce_n
    return tot / n, met


def main():
    a = parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    train_seqs, val_seqs = split_by_sequence(a.csv, a.val_frac, a.seed)
    print(f"videos: {len(train_seqs)} train / {len(val_seqs)} val   (split by SEQUENCE)")
    if a.shuffle_control:
        print("\n*** SHUFFLE CONTROL: frames paired with the WRONG gaze rates ***")
        print("    expect the model to do no better than predicting the mean\n")

    rstats = compute_channel_stats(a.csv, train_seqs)          # inputs
    tmean, tstd = compute_target_stats(a.csv, train_seqs)      # targets
    print(f"rate  channels mean {np.round(rstats[0], 4)}  std {np.round(rstats[1], 4)}")
    print(f"target mean {np.round(tmean, 4)}  std {np.round(tstd, 4)}   {list(TARGETS)}")

    mk = lambda seqs, sd: Loss2Dataset(
        a.csv, seqs, a.seq_len, rstats, root=a.feat_root, target_stats=(tmean, tstd),
        shuffle_rates=a.shuffle_control, shuffle_seed=sd)
    ds_tr = mk(train_seqs, a.seed)
    ds_va = mk(val_seqs, a.seed + 1) if val_seqs else None
    print(f"rows  : {len(ds_tr)} train" + (f" / {len(ds_va)} val" if ds_va else ""))

    sampler = VideoBalancedBatchSampler(
        ds_tr.df["sequence"].map(ds_tr.seq_to_id).to_numpy(),
        a.batch_size, a.max_per_video, a.seed)
    dl_tr = DataLoader(ds_tr, batch_sampler=sampler, num_workers=a.num_workers,
                       pin_memory=(a.device == "cuda"))
    dl_va = DataLoader(ds_va, batch_size=a.batch_size, shuffle=False,
                       num_workers=a.num_workers) if ds_va else None

    feat_dim = ds_tr[0]["z0"].numel()
    f_m = GazeRateEncoder(3, a.width, a.dim, a.dropout).to(a.device)
    head = build_head(a.dim, a.hidden, len(TARGETS), a.dropout).to(a.device)
    h = ChangePredictor(feat_dim, 1024, a.dim, a.dropout).to(a.device) if a.joint else None

    if a.init_from:
        ck = torch.load(a.init_from, map_location=a.device)
        f_m.load_state_dict(ck["f_m"])
        print(f"warm-started the gaze encoder from {a.init_from}")

    params = list(f_m.parameters()) + list(head.parameters())
    if h is not None:
        params += list(h.parameters())
    n_par = sum(p.numel() for p in params if p.requires_grad)
    print(f"trainable params {n_par:,}   objective: "
          f"{'L_sim + %.2f*L_NCE' % a.nce_weight if a.joint else 'L_sim only'}")

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    if a.dry_run:
        l, m = run_epoch(f_m, head, h, dl_tr, a, tmean, tstd)
        print(f"\ndry run OK -- loss {l:.4f}\n{format_metrics(m)}")
        return

    hist, best = [], -np.inf
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        ltr, mtr = run_epoch(f_m, head, h, dl_tr, a, tmean, tstd, opt)
        lva, mva = run_epoch(f_m, head, h, dl_va, a, tmean, tstd) if dl_va else (None, None)
        sched.step()

        line = f"ep {ep:3d}/{a.epochs}  loss {ltr:.4f}  R2 {mtr['r2_mean']:+.3f}"
        if mva:
            line += f"   |  val loss {lva:.4f}  R2 {mva['r2_mean']:+.3f}  " \
                    f"r {mva['pearson_mean']:+.3f}"
        print(line + f"   {time.time()-t0:.1f}s")

        hist.append(dict(epoch=ep, train_loss=ltr, val_loss=lva,
                         train=mtr, val=mva))
        score = mva["r2_mean"] if mva else mtr["r2_mean"]
        if score > best:
            best = score
            torch.save(dict(f_m=f_m.state_dict(), head=head.state_dict(),
                            h=(h.state_dict() if h is not None else None),
                            args=vars(a), epoch=ep, score=score,
                            target_stats=[tmean.tolist(), tstd.tolist()],
                            rate_stats=[rstats[0].tolist(), rstats[1].tolist()]),
                       os.path.join(a.out_dir, "best.pt"))

    with open(os.path.join(a.out_dir, "history.json"), "w") as fh:
        json.dump(hist, fh, indent=2)

    final = hist[-1]["val"] or hist[-1]["train"]
    print(f"\n{'='*72}\nFINAL (last epoch, {'val' if hist[-1]['val'] else 'train'} split)")
    print(format_metrics(final))
    print(f"\n   final mean R2: {final['r2_mean']:+.3f}      "
          f"best-epoch mean R2: {best:+.3f}   ->  {a.out_dir}/best.pt")
    print(f"{'='*72}")

    # The verdict deliberately uses the FINAL epoch, not the best. Taking the max over
    # 50 noisy epochs is a multiple-comparisons trap: a run with no signal at all will
    # still show a slightly positive best-epoch R2 by chance.
    if final["r2_mean"] <= 0.0:
        print("\n  R2 <= 0 means the model is NO BETTER than predicting the training\n"
              "  mean. At this scale that is the expected outcome, and it is not yet\n"
              "  evidence against the premise -- it says the experiment cannot answer\n"
              "  the question. More videos is the first thing to change.")
    else:
        print("\n  R2 > 0 on held-out VIDEOS means gaze rates carry real information\n"
              "  about how the picture changed. Check the shuffle control before\n"
              "  believing it: --shuffle_control must NOT reproduce this.")


if __name__ == "__main__":
    main()
