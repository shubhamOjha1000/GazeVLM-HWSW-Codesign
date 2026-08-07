"""Dataset for Loss 1, reading the feature CSV produced by the build notebook.

Expected columns (see notebooks/colab_build_feature_csv.ipynb):

    idx, sequence, feat_frame_1, feat_frame_2,
    gaze_patch_token_sim, frame_similarity,
    n_velocities, gaze_rates_window

Only `sequence`, the two feature paths and `gaze_rates_window` are used here; the two
similarity columns belong to Loss 2.

Nothing decodes video or runs DINOv2 -- the .npz files hold precomputed frozen features,
which is the whole reason the CSV stores feature paths rather than JPEG paths.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


def parse_rates(s):
    """'wy,wp,wm;wy,wp,wm;...' -> (L, 3) float32."""
    if not isinstance(s, str) or not s.strip():
        return np.zeros((0, 3), dtype=np.float32)
    return np.array([[float(v) for v in trip.split(",")] for trip in s.split(";")],
                    dtype=np.float32)


class Loss1Dataset(Dataset):
    """One item = one consecutive frame pair = one CSV row.

    Returns the two frozen frame features, the gaze-rate signal as (3, L) ready for
    Conv1d, and a validity mask so short windows pool correctly.
    """

    def __init__(self, csv_path, sequences=None, seq_len=9, stats=None,
                 feat_key="cls", root=None, shuffle_rates=False, shuffle_seed=0):
        df = pd.read_csv(csv_path)
        if sequences is not None:
            df = df[df["sequence"].isin(sequences)]
        self.df = df.reset_index(drop=True)
        self.seq_len = int(seq_len)
        self.feat_key = feat_key
        self.root = root                     # re-root paths if the CSV moved machines
        self.stats = stats                   # (mean(3,), std(3,)) or None

        # CONTROL: take the gaze rates from a DIFFERENT row than the frames, destroying
        # the correspondence the loss is supposed to learn. A run with this on must fail
        # to beat chance -- otherwise a real run's success is an artefact of the setup
        # (batch memorisation, a leaky split) rather than evidence of the premise.
        self.perm = None
        if shuffle_rates:
            self.perm = np.random.default_rng(shuffle_seed).permutation(len(self.df))

        seqs = sorted(self.df["sequence"].unique())
        self.seq_to_id = {s: i for i, s in enumerate(seqs)}
        self.sequences = seqs

    def __len__(self):
        return len(self.df)

    def _path(self, p):
        """Re-root a feature path, keeping the <sequence>/<file> tail.

        The basename alone is NOT unique: every sequence folder contains a
        feat_00000.npz, so re-rooting on the basename would silently load another
        video's features. The last two components are what identify a file.
        """
        if not self.root:
            return p
        parts = p.replace("\\", "/").rstrip("/").split("/")
        tail = os.path.join(*parts[-2:]) if len(parts) >= 2 else parts[-1]
        return os.path.join(self.root, tail)

    def _feat(self, p):
        with np.load(self._path(p)) as z:
            return z[self.feat_key].astype(np.float32)

    def __getitem__(self, i):
        r = self.df.iloc[i]

        z0 = self._feat(r["feat_frame_1"])
        zt = self._feat(r["feat_frame_2"])

        # normally the rates come from this same row; under the shuffle control they
        # come from another, so frames and gaze no longer describe the same moment
        rr = r if self.perm is None else self.df.iloc[int(self.perm[i])]
        rates = parse_rates(rr["gaze_rates_window"])         # (L, 3)
        if self.stats is not None:
            mean, std = self.stats
            rates = (rates - mean) / std

        # pad / truncate to a fixed length so the batch stacks; the mask keeps the
        # padding out of the pooled representation
        L = self.seq_len
        n = min(len(rates), L)
        buf = np.zeros((L, 3), dtype=np.float32)
        mask = np.zeros((L,), dtype=bool)
        if n > 0:
            buf[:n] = rates[:n]
            mask[:n] = True

        return dict(
            z0=torch.from_numpy(z0),
            zt=torch.from_numpy(zt),
            rates=torch.from_numpy(buf.T.copy()),            # (3, L) for Conv1d
            mask=torch.from_numpy(mask),
            seq_id=torch.tensor(self.seq_to_id[r["sequence"]], dtype=torch.long),
            row=torch.tensor(int(r["idx"]), dtype=torch.long),
        )


def compute_channel_stats(csv_path, sequences=None, eps=1e-6):
    """Per-channel mean/std over every step in the (training) split.

    Worth doing: omega_mag is strictly positive while the other two are zero-centred,
    and the three have different spreads. Standardising stops the magnitude channel
    dominating the first conv.
    """
    df = pd.read_csv(csv_path)
    if sequences is not None:
        df = df[df["sequence"].isin(sequences)]
    allr = [parse_rates(s) for s in df["gaze_rates_window"]]
    allr = np.concatenate([a for a in allr if len(a)], axis=0)   # (total_steps, 3)
    mean = allr.mean(axis=0).astype(np.float32)
    std = allr.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, eps)


def split_by_sequence(csv_path, val_frac=0.2, seed=0):
    """Hold out whole VIDEOS, never individual rows.

    Rows from the same video are strongly correlated -- consecutive windows during one
    fixation are near-identical -- so a random row split leaks the validation set and
    the retrieval metric becomes meaningless.
    """
    seqs = sorted(pd.read_csv(csv_path)["sequence"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n_val = max(1, int(round(len(seqs) * val_frac))) if len(seqs) > 1 else 0
    return seqs[n_val:], seqs[:n_val]                            # (train, val)


class VideoBalancedBatchSampler(Sampler):
    """Cap how many rows from one video may share a batch.

    In-batch negatives are the whole point of InfoNCE, but two windows a few seconds
    apart in the same video can have almost identical gaze rates -- false negatives the
    loss will actively push apart. Spreading each batch across videos keeps most
    negatives genuine.

    Set max_per_video >= batch_size to disable and fall back to plain shuffling.
    """

    def __init__(self, seq_ids, batch_size, max_per_video=8, seed=0, drop_last=True):
        self.batch_size = int(batch_size)
        self.max_per_video = int(max_per_video)
        self.drop_last = drop_last
        self.seed = seed
        self.by_video = {}
        for i, s in enumerate(np.asarray(seq_ids)):
            self.by_video.setdefault(int(s), []).append(i)
        self.n = len(seq_ids)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        self.seed += 1                                   # reshuffle every epoch
        pools = {v: list(rng.permutation(idx)) for v, idx in self.by_video.items()}
        batch = []
        while any(pools.values()):
            vids = [v for v, p in pools.items() if p]
            rng.shuffle(vids)
            for v in vids:
                take = min(self.max_per_video, self.batch_size - len(batch), len(pools[v]))
                for _ in range(take):
                    batch.append(int(pools[v].pop()))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if not any(pools.values()):
                break
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        return self.n // self.batch_size if self.drop_last else \
            (self.n + self.batch_size - 1) // self.batch_size
