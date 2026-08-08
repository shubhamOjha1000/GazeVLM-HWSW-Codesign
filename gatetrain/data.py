"""Fold CSV -> tensors, including the frame features Loss 1 needs.

Two things are loaded, and they have very different costs.

The gaze side is cheap: the rate columns parse into a (N, 3, 9) array, ~1.4 MB, which
goes straight onto the device.

The frame side is not. Loss 1 (PDF p2) needs z_I0 and z_IT, the frozen DINOv2 CLS tokens,
which live in one .npz per frame on Drive. Reading two per row per epoch would dominate
every other cost in the job. So `FeatureCache` reads each UNIQUE frame exactly once into a
(n_frames, 384) matrix -- about 20 MB for 13,000 frames -- and rows then index into it.
Drive is paid once per session rather than once per step, and the matrix is optionally
saved to local disk so a rerun skips even that.

The five fold files contain the same rows in the same order and differ only in `split`, so
the cache is built once and shared across all folds.
"""

import json
import os

import numpy as np
import pandas as pd
import torch

from src.loss1.dataset import parse_rates

RATE_COLS = ("gaze_rates_window", "gaze_vec3d_rates_window")
TARGETS = ("frame_similarity", "gaze_patch_token_sim")
QUAD_NAMES = {0: "TRANSITION", 1: "PURSUIT", 2: "REFIXATION", 3: "STABLE"}


def reroot(p, root):
    """Re-root a feature path, keeping the <sequence>/<file> tail.

    The basename alone is NOT unique -- every sequence folder holds a feat_00000.npz -- so
    re-rooting on it would silently load another video's features.
    """
    if not root:
        return p
    parts = str(p).replace("\\", "/").rstrip("/").split("/")
    tail = os.path.join(*parts[-2:]) if len(parts) >= 2 else parts[-1]
    return os.path.join(root, tail)


class FeatureCache:
    """Unique frame path -> row of a (n_frames, feat_dim) matrix."""

    def __init__(self, paths, feat_root=None, key="cls", cache_file=None, verbose=True):
        uniq = sorted(set(paths))
        self.index = {p: i for i, p in enumerate(uniq)}

        if cache_file and os.path.exists(cache_file):
            z = np.load(cache_file, allow_pickle=True)
            if list(z["paths"]) == uniq:
                self.F = z["F"].astype(np.float32)
                if verbose:
                    print(f"   feature cache hit: {self.F.shape} from {cache_file}")
                return
            if verbose:
                print("   feature cache exists but does not match these rows; rebuilding")

        first = np.load(reroot(uniq[0], feat_root))[key]
        self.F = np.zeros((len(uniq), first.shape[-1]), dtype=np.float32)
        miss = []
        for i, p in enumerate(uniq):
            fp = reroot(p, feat_root)
            try:
                with np.load(fp) as z:
                    self.F[i] = z[key].astype(np.float32)
            except Exception:
                miss.append(fp)
            if verbose and (i + 1) % 2000 == 0:
                print(f"   loaded {i+1:,}/{len(uniq):,} frame features")
        if miss:
            raise FileNotFoundError(
                f"{len(miss)} feature files could not be read, e.g. {miss[:3]}. "
                f"Pass --feat_root pointing at the folder that holds <sequence>/feat_*.npz, "
                f"or run with --sim_only to train Loss 2 alone.")
        if verbose:
            print(f"   loaded {len(uniq):,} unique frame features -> {self.F.shape}")
        if cache_file:
            os.makedirs(os.path.dirname(os.path.abspath(cache_file)) or ".", exist_ok=True)
            np.savez_compressed(cache_file, F=self.F, paths=np.array(uniq))
            if verbose:
                print(f"   cached -> {cache_file}")

    @property
    def dim(self):
        return int(self.F.shape[1])

    def rows(self, paths):
        return np.array([self.index[p] for p in paths], dtype=np.int64)


def pack_rates(series, seq_len):
    """'a,b,c;...' x N -> (N, 3, L) float32 and a (N, L) bool mask.

    Padded steps are masked rather than zeroed and forgotten: zero is a legitimate angular
    rate, so pooling has to know which steps are real.
    """
    n = len(series)
    X = np.zeros((n, seq_len, 3), dtype=np.float32)
    M = np.zeros((n, seq_len), dtype=bool)
    for i, s in enumerate(series):
        r = parse_rates(s)
        k = min(len(r), seq_len)
        if k:
            X[i, :k] = r[:k]
            M[i, :k] = True
    return np.ascontiguousarray(X.transpose(0, 2, 1)), M


def channel_stats(X, M, eps=1e-6):
    """Per-channel mean/std over REAL steps only -- padding would drag both toward zero
    by an amount that depends on how many short windows the split happens to contain."""
    real = X.transpose(0, 2, 1)[M]
    return real.mean(0).astype(np.float32), np.maximum(real.std(0), eps).astype(np.float32)


class FoldSet:
    """One fold CSV as device-resident tensors.

    Holds: the gaze rates, the two regression targets (raw AND standardised), indices into
    the shared feature cache, and the label columns needed for the threshold consistency
    check.
    """

    def __init__(self, csv_path, cache=None, rates_col="gaze_rates_window", seq_len=9,
                 device="cpu", rate_stats=None, target_stats=None, feat_root=None):
        if rates_col not in RATE_COLS:
            raise ValueError(f"rates_col must be one of {RATE_COLS}, got {rates_col!r}")

        have = set(pd.read_csv(csv_path, nrows=0).columns)
        need = ["sequence", rates_col, *TARGETS, "quad"]
        missing = [c for c in need if c not in have]
        if missing:
            raise KeyError(f"{os.path.basename(csv_path)} is missing {missing}. "
                           f"Was it written by notebooks/colab_label_quadrants.ipynb?")
        cols = need + [c for c in ("split", "gate", "feat_frame_1", "feat_frame_2")
                       if c in have]
        df = pd.read_csv(csv_path, usecols=cols)

        self.df = df
        self.device = device
        self.rates_col = rates_col
        self.csv_path = csv_path

        X, M = pack_rates(df[rates_col].tolist(), seq_len)
        self.rate_stats = rate_stats if rate_stats is not None else channel_stats(X, M)
        rm, rs = self.rate_stats
        X = (X - rm[None, :, None]) / rs[None, :, None]

        y_raw = df[list(TARGETS)].to_numpy(dtype=np.float32)
        # standardised because the two targets have different spreads (~0.13 vs ~0.22);
        # raw MSE would weight the gaze term ~3x for no principled reason
        self.target_stats = (target_stats if target_stats is not None
                             else (y_raw.mean(0).astype(np.float32),
                                   np.maximum(y_raw.std(0), 1e-6).astype(np.float32)))
        tm, ts = self.target_stats

        self.X = torch.from_numpy(X).to(device)
        self.M = torch.from_numpy(M).to(device)
        self.y_raw_np = y_raw
        self.y = torch.from_numpy((y_raw - tm) / ts).to(device)

        self.quad_np = df["quad"].to_numpy()
        self.gate_np = (df["gate"].to_numpy() if "gate" in df.columns else None)
        self.split = df["split"].to_numpy() if "split" in df.columns else None
        self.sequence = df["sequence"].to_numpy()

        self.i0 = self.iT = None
        if cache is not None and "feat_frame_1" in df.columns:
            self.i0 = torch.from_numpy(cache.rows(df["feat_frame_1"])).to(device)
            self.iT = torch.from_numpy(cache.rows(df["feat_frame_2"])).to(device)

    # ------------------------------------------------------------------ views
    def subset(self, mask):
        idx = np.flatnonzero(np.asarray(mask))
        v = object.__new__(FoldSet)
        v.__dict__.update(self.__dict__)
        t = torch.from_numpy(idx).to(self.device)
        v.X, v.M, v.y = self.X[t], self.M[t], self.y[t]
        v.y_raw_np = self.y_raw_np[idx]
        v.quad_np = self.quad_np[idx]
        v.gate_np = None if self.gate_np is None else self.gate_np[idx]
        v.sequence = self.sequence[idx]
        v.split = None if self.split is None else self.split[idx]
        v.df = self.df.iloc[idx].reset_index(drop=True)
        v.i0 = None if self.i0 is None else self.i0[t]
        v.iT = None if self.iT is None else self.iT[t]
        return v

    def train_val(self):
        if self.split is None:
            raise KeyError(f"{os.path.basename(self.csv_path)} has no `split` column "
                           f"-- that file is the test set, not a fold")
        return self.subset(self.split == "train"), self.subset(self.split == "val")

    def __len__(self):
        return int(self.X.shape[0])

    # ------------------------------------------------------------------ batching
    def batches(self, batch_size, shuffle=False, generator=None, max_per_video=None,
                seq_ids=None):
        """Index batches. `max_per_video` caps how many rows of one video share a batch.

        That cap exists for InfoNCE: negatives come from the batch, and two windows a few
        seconds apart in the same video have near-identical gaze rates. As negatives they
        are false ones, and the loss would actively push them apart.
        """
        n = len(self)
        if shuffle and max_per_video and seq_ids is not None:
            # yields the batches AS BUILT. Returning a flat order and re-slicing at fixed
            # strides would break the cap: short batches shift every later boundary, so
            # the batches the trainer sees would not be the ones the sampler balanced.
            for b in _balanced_batches(seq_ids, batch_size, max_per_video, generator):
                yield torch.from_numpy(b).to(self.device)
            return
        idx = (torch.randperm(n, device=self.device, generator=generator)
               if shuffle else torch.arange(n, device=self.device))
        for s in range(0, n, batch_size):
            yield idx[s:s + batch_size]

    def gather(self, j, features=None):
        z0 = zT = None
        if features is not None and self.i0 is not None:
            z0, zT = features[self.i0[j]], features[self.iT[j]]
        return self.X[j], self.M[j], self.y[j], z0, zT

    def video_ids(self):
        u = {s: i for i, s in enumerate(sorted(set(self.sequence)))}
        return np.array([u[s] for s in self.sequence], dtype=np.int64)


def _balanced_batches(seq_ids, batch_size, max_per_video, generator=None):
    """Build batches in which no batch holds more than `max_per_video` rows of one video.

    Why the cap: InfoNCE draws its negatives from the batch, and two windows a few seconds
    apart in the same video have near-identical gaze. As negatives they are false ones,
    and the loss actively pushes them apart.

    The count is tracked PER BATCH and reset when a batch closes. An earlier version
    tracked nothing and simply took up to `max_per_video` from each video on every pass
    over the pool; late in an epoch, when only a few videos still had rows, the outer loop
    re-entered and topped up the same partly-filled batch from the same video again and
    again -- producing batches with 24 rows of one video under a cap of 8.

    When the cap makes a full batch impossible -- one video left, say -- a SHORT batch is
    emitted rather than the cap being broken. Fewer negatives is a smaller problem than
    false ones, and `joint_loss` already skips the NCE term below two rows.
    """
    rng = np.random.default_rng(
        int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        if generator is not None else 0)
    pools = {}
    for i, v in enumerate(np.asarray(seq_ids)):
        pools.setdefault(int(v), []).append(i)
    for v in pools:
        pools[v] = list(rng.permutation(pools[v]))

    batches = []
    while any(pools.values()):
        batch, used = [], {}
        added = True
        while len(batch) < batch_size and added:
            added = False
            vids = [v for v, p in pools.items()
                    if p and used.get(v, 0) < max_per_video]
            rng.shuffle(vids)
            for v in vids:
                if len(batch) >= batch_size:
                    break
                batch.append(int(pools[v].pop()))
                used[v] = used.get(v, 0) + 1
                added = True
        if not batch:
            break
        batches.append(np.array(batch, dtype=np.int64))
    return batches


def train_stats(csv_path, rates_col="gaze_rates_window", seq_len=9):
    """Channel and target statistics from the TRAIN rows of a fold file, and only those.

    Computed here rather than by subsetting a FoldSet: `subset()` copies the parent's
    attribute dict, so a subset inherits whatever statistics the parent was built with.
    Taking `.rate_stats` off a train subset of a whole-file FoldSet therefore returns
    WHOLE-FILE statistics -- val rows leaking into the normalisation, quietly, with
    nothing in the output to show for it.
    """
    df = pd.read_csv(csv_path, usecols=["split", rates_col, *TARGETS])
    tr = df[df["split"] == "train"]
    if not len(tr):
        raise ValueError(f"{os.path.basename(csv_path)} has no rows with split == 'train'")
    X, M = pack_rates(tr[rates_col].tolist(), seq_len)
    y = tr[list(TARGETS)].to_numpy(dtype=np.float32)
    return (channel_stats(X, M),
            (y.mean(0).astype(np.float32), np.maximum(y.std(0), 1e-6).astype(np.float32)))


def load_thresholds(path):
    """The τ_frame / τ_gaze that produced the labels, from the labelling notebook."""
    with open(path) as fh:
        t = json.load(fh)
    for k in ("tau_frame", "tau_gaze"):
        if k not in t:
            raise KeyError(f"{path} has no {k!r}; keys are {list(t)}")
    return float(t["tau_frame"]), float(t["tau_gaze"]), t


def check_thresholds(y_raw, quad, tau_f, tau_g, name=""):
    """Applying the saved thresholds to the TRUE similarities must reproduce `quad`.

    This is the cheap check that the thresholds.json in Drive actually belongs to these
    fold files. If someone re-ran the labelling notebook with different percentiles and
    did not rebuild the folds, every downstream number would be quietly measured against
    the wrong decision boundary.
    """
    recomputed = 2 * (y_raw[:, 0] > tau_f).astype(int) + (y_raw[:, 1] > tau_g).astype(int)
    agree = float((recomputed == np.asarray(quad)).mean())
    ok = agree > 0.999
    print(f"   threshold check {name}: saved tau reproduces `quad` on "
          f"{100*agree:.2f}% of rows   [{'PASS' if ok else 'FAIL'}]")
    if not ok:
        raise ValueError(
            f"thresholds.json does not match {name}: applying tau_frame={tau_f:.4f} / "
            f"tau_gaze={tau_g:.4f} to the true similarities reproduces only "
            f"{100*agree:.1f}% of the stored quad labels. Re-run the labelling notebook "
            f"or point --thresholds at the file that produced these folds.")
    return agree
