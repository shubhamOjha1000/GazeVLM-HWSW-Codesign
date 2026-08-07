"""Loss-1 pretraining loop.

Hyperparameter defaults are EgoDistill Sec 4.1's pretraining settings: AdamW, lr 1e-4,
batch 64, 50 epochs, tau = 0.1.

    python -m src.loss1.train --csv /content/build/feature_dataset.csv \
                              --out_dir runs/loss1

Note what is NOT here: EgoDistill's L_pretrain is L_NCE + L1, where L1 matches a heavy
video model's clip feature. This project has no video model, so only L_NCE is used. The
grounding role L1 played is served later by Loss 2, which regresses the DINOv2
similarities already sitting in the CSV.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.loss1.dataset import (Loss1Dataset, VideoBalancedBatchSampler,
                               compute_channel_stats, split_by_sequence)
from src.loss1.losses import info_nce, retrieval_metrics
from src.loss1.models import build_models, count_params


def parse_args():
    ap = argparse.ArgumentParser("Loss-1 (InfoNCE) pretraining for the gaze-rate encoder")
    ap.add_argument("--csv", required=True, help="feature CSV from the build notebook")
    ap.add_argument("--out_dir", default="runs/loss1")
    ap.add_argument("--feat_root", default=None,
                    help="re-root the .npz paths if the CSV was written elsewhere")

    ap.add_argument("--epochs", type=int, default=50)         # EgoDistill 4.1
    ap.add_argument("--batch_size", type=int, default=64)     # EgoDistill 4.1
    ap.add_argument("--lr", type=float, default=1e-4)         # EgoDistill 4.1
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--tau", type=float, default=0.1)         # EgoDistill Eq. 7

    ap.add_argument("--dim", type=int, default=256, help="shared embedding size")
    ap.add_argument("--hidden", type=int, default=1024)       # EgoDistill fusion width
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seq_len", type=int, default=9,
                    help="steps per window; 9 for a 1 s window at 10 Hz")
    ap.add_argument("--rates_col", default="gaze_rates_window",
                    choices=["gaze_rates_window", "gaze_vec3d_rates_window"],
                    help="which 9x3 signal feeds f_M. gaze_vec3d_rates_window is the "
                         "same shape but sphere-exact: its norm is the true angular "
                         "speed, while omega_mag over-reads by 1/cos(pitch)")

    ap.add_argument("--val_frac", type=float, default=0.2, help="fraction of VIDEOS held out")
    ap.add_argument("--max_per_video", type=int, default=8,
                    help="cap rows from one video per batch; >= batch_size disables")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry_run", action="store_true", help="two batches, then stop")
    ap.add_argument("--shuffle_control", action="store_true",
                    help="CONTROL: pair each frame pair with ANOTHER row's gaze rates. "
                         "Must fail to beat chance; if it succeeds, a real run's result "
                         "is an artefact rather than evidence.")
    return ap.parse_args()


def run_epoch(f_m, h, loader, tau, device, opt=None):
    train = opt is not None
    f_m.train(train); h.train(train)
    tot, n, agg = 0.0, 0, dict(top1=0.0, top5=0.0, mean_rank=0.0, chance=0.0)

    for batch in loader:
        z0 = batch["z0"].to(device, non_blocking=True)
        zt = batch["zt"].to(device, non_blocking=True)
        rates = batch["rates"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        if z0.size(0) < 2:            # InfoNCE needs at least one negative
            continue

        with torch.set_grad_enabled(train):
            z_hat = h(z0, zt)                     # from FROZEN frame features
            z_m = f_m(rates, mask)                # from the gaze-rate signal
            loss, logits = info_nce(z_hat, z_m, tau)

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(f_m.parameters()) + list(h.parameters()), 5.0)
            opt.step()

        m = retrieval_metrics(logits.detach())
        bs = z0.size(0)
        tot += loss.item() * bs; n += bs
        for k in agg:
            agg[k] += m[k] * bs

    if n == 0:
        return dict(loss=float("nan"), top1=0.0, top5=0.0, mean_rank=0.0, chance=0.0)
    out = {k: v / n for k, v in agg.items()}
    out["loss"] = tot / n
    return out


def main():
    a = parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    train_seqs, val_seqs = split_by_sequence(a.csv, a.val_frac, a.seed)
    print(f"videos: {len(train_seqs)} train / {len(val_seqs)} val   (split by SEQUENCE)")
    if not val_seqs:
        print("  !! only one video available -- no held-out split, val metrics disabled")

    # statistics from the TRAIN split only, so the val split stays untouched
    stats = compute_channel_stats(a.csv, train_seqs, rates_col=a.rates_col)
    print(f"input signal: {a.rates_col}")
    print(f"rate channel mean {np.round(stats[0], 4)}  std {np.round(stats[1], 4)}")

    if a.shuffle_control:
        print("\n*** SHUFFLE CONTROL: frames paired with the WRONG gaze rates ***")
        print("    expect loss to stay near ln(batch) and top-1 near chance\n")
    ds_tr = Loss1Dataset(a.csv, train_seqs, a.seq_len, stats, root=a.feat_root,
                         shuffle_rates=a.shuffle_control, shuffle_seed=a.seed,
                         rates_col=a.rates_col)
    ds_va = Loss1Dataset(a.csv, val_seqs, a.seq_len, stats, root=a.feat_root,
                         shuffle_rates=a.shuffle_control, shuffle_seed=a.seed + 1,
                         rates_col=a.rates_col) \
        if val_seqs else None
    print(f"rows  : {len(ds_tr)} train" + (f" / {len(ds_va)} val" if ds_va else ""))

    sampler = VideoBalancedBatchSampler(
        ds_tr.df["sequence"].map(ds_tr.seq_to_id).to_numpy(),
        a.batch_size, a.max_per_video, a.seed)
    dl_tr = DataLoader(ds_tr, batch_sampler=sampler, num_workers=a.num_workers,
                       pin_memory=(a.device == "cuda"))
    dl_va = DataLoader(ds_va, batch_size=a.batch_size, shuffle=False, drop_last=False,
                       num_workers=a.num_workers) if ds_va else None

    probe = ds_tr[0]
    f_m, h = build_models(feat_dim=probe["z0"].numel(), dim=a.dim, hidden=a.hidden,
                          width=a.width, dropout=a.dropout)
    f_m, h = f_m.to(a.device), h.to(a.device)
    print(f"feature dim {probe['z0'].numel()} -> embedding {a.dim}   "
          f"trainable params {count_params(f_m, h):,}")
    print("   (the image encoder is NOT trained: features are precomputed and frozen,\n"
          "    which EgoDistill footnote 2 reports is required to avoid mode collapse)")

    opt = torch.optim.AdamW(list(f_m.parameters()) + list(h.parameters()),
                            lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    if a.dry_run:
        b = next(iter(dl_tr))
        z_hat = h(b["z0"].to(a.device), b["zt"].to(a.device))
        z_m = f_m(b["rates"].to(a.device), b["mask"].to(a.device))
        loss, logits = info_nce(z_hat, z_m, a.tau)
        print(f"\ndry run OK -- batch {b['z0'].shape[0]}, loss {loss.item():.4f}, "
              f"{retrieval_metrics(logits)}")
        return

    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr = run_epoch(f_m, h, dl_tr, a.tau, a.device, opt)
        va = run_epoch(f_m, h, dl_va, a.tau, a.device) if dl_va else None
        sched.step()

        line = (f"ep {ep:3d}/{a.epochs}  loss {tr['loss']:.4f}  "
                f"top1 {tr['top1']*100:5.1f}%  (chance {tr['chance']*100:4.1f}%)")
        if va:
            line += f"   |  val loss {va['loss']:.4f}  top1 {va['top1']*100:5.1f}%"
        print(line + f"   {time.time()-t0:.1f}s")

        hist.append(dict(epoch=ep, train=tr, val=va))
        score = va["top1"] if va else tr["top1"]
        if score > best:
            best = score
            torch.save(dict(f_m=f_m.state_dict(), h=h.state_dict(),
                            args=vars(a), stats=[stats[0].tolist(), stats[1].tolist()],
                            epoch=ep, score=score),
                       os.path.join(a.out_dir, "best.pt"))

    with open(os.path.join(a.out_dir, "history.json"), "w") as fh:
        json.dump(hist, fh, indent=2)
    print(f"\nbest retrieval top-1: {best*100:.1f}%  ->  {a.out_dir}/best.pt")
    print("Compare against chance (1/batch). Well above chance means gaze rates ARE\n"
          "predictable from visual change -- the premise the whole project rests on.")


if __name__ == "__main__":
    main()
