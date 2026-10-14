"""Eye + head features for the 4-way gate: TRANSITION / PURSUIT / REFIXATION / STABLE.

Three per-step signals, all on the eye tracker's clock (~10 Hz, so 9 steps per 1 s window):

    eye_speed(k)  = sqrt(eye_yaw_rate^2  + eye_pitch_rate^2)                    rad/s
    head_speed(k) = sqrt(head_yaw_rate^2 + head_pitch_rate^2)                   rad/s
    cos(k)        = (e . h) / (|e| |h|),  e, h = (yaw rate, pitch rate)         [-1, 1]

cos is +1 when eye and head turn the same way (a gaze shift) and -1 when the eyes
counter-rotate against the head (the vestibulo-ocular reflex holding a target). It is set
to 0 when either speed is below a noise floor, where the direction is meaningless.

Where each signal comes from
----------------------------
EYE   `gaze_rates_window`, columns 0-1: d(yaw)/dt, d(pitch)/dt of the MPS gaze, which is
      expressed in the Central Pupil Frame (CPF: +X left, +Y up, +Z forward). Following
      `gaze_geometry.yawpitch_to_unit_vec`, +yaw turns the gaze toward +X (LEFT) and
      +pitch toward +Y (UP).

HEAD  `imu_window`, columns 3-5: gyro x/y/z in rad/s, binned to ~10 rows per window, in
      the IMU's OWN sensor frame (right IMU, stream 1202-1). Before yaw/pitch mean
      anything the gyro must be put in the eye's convention. In CPF, by the right-hand
      rule, +w_y turns the forward axis toward +X (left) and +w_x turns it toward -Y
      (down). So, for a rotation R_cpf<-imu,

          head_yaw   =  (R w)_y
          head_pitch = -(R w)_x

      which is the 2x3 matrix A = [R[1]; -R[0]] applied to the raw gyro. With that sign
      convention a perfect VOR gives eye = -head, i.e. cos = -1.

Three ways to get A (`--imu_rot`):

    calib  from Aria's factory calibration (eyehead/imu_calib.py writes the JSON). Exact.
    fit    learned from the TRAIN rows' inputs alone: during ordinary looking the eyes
           mostly counter-rotate against the head, so eye ~= -A w by least squares,
           orthonormalised. No labels are touched, but it assumes VOR dominates, and it
           makes the VOR sign check below circular.
    none   assume the IMU axes already equal CPF. Unverified -- a diagnostic only.

Alignment: the IMU arrives as ~10 equal-width bins over the frame interval, the eye as
L = n-1 steps between ~10 samples. Bin b sits at (b + 0.5)/n_bins of the window and step k
at (k + 1)/(L + 1); the gyro is linearly interpolated onto the step centres. That is an
approximation -- the CSV stores no per-sample timestamps -- good to roughly half a step.
"""

import json

import numpy as np
import pandas as pd

from src.loss1.dataset import parse_rates

EYE_COL = "gaze_rates_window"
IMU_COL = "imu_window"
TARGETS = ("frame_similarity", "gaze_patch_token_sim")
NAMES = {0: "TRANSITION", 1: "PURSUIT", 2: "REFIXATION", 3: "STABLE"}
CHANNELS = ("eye_speed", "head_speed", "cos_eye_head")
STREAM_CH = {"eye": 0, "head": 1, "cos": 2}

# IMU axes assumed equal to CPF -- `--imu_rot none`
A_ASSUME_CPF = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------- loading
def load_rows(csv_path):
    """The columns this package needs, nothing else -- the packed strings are most of
    the file, and the DINOv2 feature paths are not used at all."""
    have = set(pd.read_csv(csv_path, nrows=0).columns)
    need = ["sequence", EYE_COL, "quad", *TARGETS]
    missing = [c for c in need if c not in have]
    if missing:
        raise KeyError(f"{csv_path} is missing {missing}. "
                       "Was it written by notebooks/colab_label_quadrants.ipynb?")
    if IMU_COL not in have:
        raise KeyError(f"{csv_path} has no {IMU_COL!r}. It was built with --no_imu; "
                       "rebuild it with src.dataprep.build_feature_csv.")
    cols = need + [IMU_COL] + [c for c in ("split", "n_imu_in_gap") if c in have]
    return pd.read_csv(csv_path, usecols=cols)


def head_map_from_rotation(R):
    """3x3 R_cpf<-imu -> the 2x3 map from raw gyro to (head_yaw, head_pitch)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return np.stack([R[1], -R[0]])


def load_imu_calib(path):
    c = json.load(open(path))
    if "R_cpf_imu" not in c:
        raise KeyError(f"{path} has no 'R_cpf_imu'; write it with eyehead.imu_calib")
    return head_map_from_rotation(c["R_cpf_imu"]), c


# ---------------------------------------------------------------------------- raw signals
def _gyro_at_steps(imu_rows, L):
    """(n_bins, 6) binned IMU -> (L, 3) gyro at the gaze-step centres, or None.

    Empty bins are NaN in the CSV (zero is a real reading, so the builder never writes
    zero for "missing"); they are skipped rather than interpolated through as zeros.
    """
    if imu_rows.ndim != 2 or imu_rows.shape[0] == 0 or imu_rows.shape[1] < 6:
        return None
    g = imu_rows[:, 3:6].astype(np.float64)
    ok = np.isfinite(g).all(axis=1)
    if not ok.any():
        return None
    centres = (np.arange(len(g)) + 0.5) / len(g)
    steps = (np.arange(L) + 1.0) / (L + 1.0)
    return np.stack([np.interp(steps, centres[ok], g[ok, j]) for j in range(3)], axis=1)


def raw_signals(df, seq_len=9):
    """Per row: eye rates (N, L, 2), gyro on the same steps (N, L, 3), step mask (N, L),
    and imu_ok (N,) -- False where the window had no usable IMU."""
    n = len(df)
    E = np.zeros((n, seq_len, 2))
    G = np.zeros((n, seq_len, 3))
    M = np.zeros((n, seq_len), dtype=bool)
    imu_ok = np.zeros(n, dtype=bool)
    for i, (es, im) in enumerate(zip(df[EYE_COL].tolist(), df[IMU_COL].tolist())):
        r = parse_rates(es)
        L = min(len(r), seq_len)
        if L == 0:
            continue
        E[i, :L] = r[:L, :2]
        M[i, :L] = True
        if isinstance(im, str) and im.strip():
            g = _gyro_at_steps(parse_rates(im), L)
            if g is not None:
                G[i, :L] = g
                imu_ok[i] = True
    return E, G, M, imu_ok


# ---------------------------------------------------------------------------- head map
def fit_head_map(E, G, M, imu_ok, head_min=0.3, eye_max=1.0):
    """Estimate A from inputs alone, assuming the eyes mostly counter-rotate the head.

    Uses steps where the head is clearly turning (|gyro| > head_min) and the eye is NOT
    making a saccade (|eye| < eye_max) -- during those, eye ~= -A w. Least squares gives
    the 2x3 map including the VOR gain; the nearest matrix with orthonormal rows (polar
    decomposition) keeps the rotation and drops the gain, so head speed stays in rad/s.
    """
    sel = M & imu_ok[:, None]
    sel &= (np.linalg.norm(G, axis=-1) > head_min) & (np.linalg.norm(E, axis=-1) < eye_max)
    g, e = G[sel], E[sel]
    if len(g) < 200:
        raise ValueError(f"only {len(g)} head-turning, non-saccadic steps to fit the IMU "
                         f"map from; lower head_min or use --imu_rot calib")
    Mt, *_ = np.linalg.lstsq(g, -e, rcond=None)          # g @ Mt ~= -e,  Mt is (3, 2)
    U, S, Vt = np.linalg.svd(Mt.T, full_matrices=False)  # (2,2), (2,), (2,3)
    A = U @ Vt
    resid = e + g @ Mt
    r2 = 1.0 - float((resid ** 2).sum() / ((e - e.mean(0)) ** 2).sum())
    return A, dict(n_steps=int(len(g)), vor_gain=S.tolist(), r2=r2,
                   A_raw=Mt.T.tolist(), A=A.tolist())


# ---------------------------------------------------------------------------- channels
def compute_channels(E, G, M, A, eye_min=0.1, head_min=0.1):
    """-> X (N, 3, L) = [eye_speed, head_speed, cos], plus the head rates (N, L, 2)."""
    H = G @ np.asarray(A).T
    es = np.linalg.norm(E, axis=-1)
    hs = np.linalg.norm(H, axis=-1)
    active = (es > eye_min) & (hs > head_min)
    cos = np.where(active, (E * H).sum(-1) / np.maximum(es * hs, 1e-9), 0.0)
    X = np.stack([es, hs, cos], axis=1) * M[:, None, :]
    return X.astype(np.float32), H


def fit_stats(X, M):
    """Per-channel mean/std over REAL steps, after log1p on the two speeds.

    Speeds are heavy-tailed (a saccade step is 10-50x a fixation step), so they are
    log-compressed before standardising; cos is already in [-1, 1] and left alone.
    """
    Z = _log_speeds(X)
    real = Z.transpose(0, 2, 1)[M]                 # (n_real_steps, 3)
    mean = real.mean(0)
    std = np.maximum(real.std(0), 1e-6)
    mean[2], std[2] = 0.0, 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalise(X, M, stats):
    mean, std = (np.asarray(s, np.float32) for s in stats)
    Z = (_log_speeds(X) - mean[None, :, None]) / std[None, :, None]
    return (Z * M[:, None, :]).astype(np.float32)


def _log_speeds(X):
    Z = X.astype(np.float32).copy()
    Z[:, :2] = np.log1p(np.maximum(Z[:, :2], 0.0))
    return Z


# ---------------------------------------------------------------------------- diagnostics
def vor_check(E, H, M, imu_ok, head_min=0.3, eye_min=0.1, eye_max=1.0):
    """Sign check on the head map: when the head turns and the eye moves smoothly (no
    saccade), the eyes should mostly counter-rotate, so cos should be strongly negative.
    A positive median means a sign or axis in A is wrong."""
    es, hs = np.linalg.norm(E, axis=-1), np.linalg.norm(H, axis=-1)
    sel = M & imu_ok[:, None] & (hs > head_min) & (es > eye_min) & (es < eye_max)
    c = (E[sel] * H[sel]).sum(-1) / np.maximum(es[sel] * hs[sel], 1e-9)
    if not len(c):
        return dict(n=0, median_cos=float("nan"), frac_negative=float("nan"))
    return dict(n=int(len(c)), median_cos=float(np.median(c)),
                frac_negative=float((c < 0).mean()))


def class_profile(X, M, y):
    """What each true class looks like in the three raw channels -- the table to read
    before training. Means are over real steps; cos statistics over active steps only."""
    rows = []
    for c in range(4):
        r = y == c
        if not r.any():
            continue
        m = M[r]
        es, hs, cs = X[r, 0][m], X[r, 1][m], X[r, 2][m]
        act = cs != 0
        rows.append(dict(
            cls=NAMES[c], rows=int(r.sum()),
            eye_speed_mean=float(es.mean()),
            eye_speed_max=float(np.where(m, X[r, 0], 0).max(1).mean()),
            head_speed_mean=float(hs.mean()),
            frac_active_steps=float(act.mean()),
            cos_mean_active=float(cs[act].mean()) if act.any() else float("nan"),
            frac_counter=float((cs < -0.5).mean()),
            frac_same_dir=float((cs > 0.5).mean()),
        ))
    return pd.DataFrame(rows).set_index("cls")


# ---------------------------------------------------------------------------- one call
def build(df, A, seq_len=9, eye_min=0.1, head_min=0.1):
    """df -> raw channels X (N,3,L), mask M (N,L), imu_ok (N,), eye/head rates for checks.

    Rows with no gaze step at all get one dummy step, so masked max-pooling stays defined
    and the row order matches the CSV (needed to compare against other gates' outputs).
    """
    E, G, M, imu_ok = raw_signals(df, seq_len)
    no_gaze = ~M.any(1)
    M[no_gaze, 0] = True
    X, H = compute_channels(E, G, M, A, eye_min, head_min)
    return dict(X=X, M=M, imu_ok=imu_ok, no_gaze=no_gaze, E=E, G=G, H=H)
