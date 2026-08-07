import os, glob, json, zipfile, urllib.request, cv2
import numpy as np, pandas as pd


def download_sequence(seq, meta, raw_dir, want_calibration=False):
    """Download RGB video + eye-gaze (and optionally the VRS for calibration + IMU).
    Returns (seq_dir, mp4_path, vrs_path). vrs_path is None when not requested."""
    seq_dir = os.path.join(raw_dir, seq)
    os.makedirs(os.path.join(seq_dir, "eye_gaze"), exist_ok=True)
    e = meta["sequences"][seq]

    rgb = e["video_main_rgb"]
    mp4 = os.path.join(seq_dir, rgb["filename"])
    if not os.path.exists(mp4):
        urllib.request.urlretrieve(rgb["download_url"], mp4)

    eg = e["mps_eye_gaze"]
    zp = os.path.join(seq_dir, eg["filename"])
    if not os.path.exists(zp):
        urllib.request.urlretrieve(eg["download_url"], zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(os.path.join(seq_dir, "eye_gaze"))

    # VRS carries the device calibration AND the IMU stream
    vrs_path = None
    if want_calibration:
        vrs = e.get("main_vrs")
        if vrs is None:
            raise RuntimeError(
                f"[{seq}] calibration/IMU requested but no 'main_vrs' entry in the URLs JSON.")
        vrs_path = os.path.join(seq_dir, vrs["filename"])
        if not os.path.exists(vrs_path):
            urllib.request.urlretrieve(vrs["download_url"], vrs_path)
        if not os.path.exists(vrs_path):
            raise RuntimeError(f"[{seq}] VRS download failed; no file at {vrs_path}")

    return seq_dir, mp4, vrs_path


def load_gaze_raw(seq_dir):
    """RAW high-rate gaze (do NOT downsample) -> (ts_us, yaw, pitch)."""
    gcsv = glob.glob(os.path.join(seq_dir, "eye_gaze", "**", "general_eye_gaze.csv"),
                     recursive=True)[0]
    g = pd.read_csv(gcsv)
    ts = g["tracking_timestamp_us"].to_numpy(); ts = ts - ts[0]
    return ts, g["yaw_rads_cpf"].to_numpy(), g["pitch_rads_cpf"].to_numpy()


def load_imu_from_vrs(vrs_path):
    """Extract the IMU stream from the VRS via projectaria_tools (like the calibration).
    Returns (ts_us, imu_array[N,6]) = [accel_xyz, gyro_xyz], or (None, None) on failure.
    NOTE: projectaria_tools stream-ID / accessor names vary by version -- verify/adjust."""
    if vrs_path is None or not os.path.exists(vrs_path):
        return None, None
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.stream_id import StreamId
    except Exception as ex:
        print(f"  [IMU] projectaria_tools import failed: {ex}")
        return None, None

    provider = data_provider.create_vrs_data_provider(vrs_path)

    # Aria has two IMU streams (1202-1 right, 1202-2 left). Prefer whichever exists.
    imu_stream = None
    for sid in ("1202-1", "1202-2"):
        try:
            s = StreamId(sid)
            if provider.get_num_data(s) > 0:
                imu_stream = s
                break
        except Exception:
            continue
    if imu_stream is None:
        print("  [IMU] no IMU stream found in VRS")
        return None, None

    n = provider.get_num_data(imu_stream)
    ts, rows = [], []
    for i in range(n):
        rec = provider.get_imu_data_by_index(imu_stream, i)
        t_ns = _field(rec, "capture_timestamp_ns")
        acc = _field(rec, "accel_msec2")
        gyr = _field(rec, "gyro_radsec")
        if t_ns is None or acc is None or gyr is None:
            continue
        ts.append(float(t_ns) / 1e3)  # ns -> us
        rows.append([float(acc[0]), float(acc[1]), float(acc[2]),
                     float(gyr[0]), float(gyr[1]), float(gyr[2])])
    if not ts:
        return None, None
    ts = np.array(ts); ts = ts - ts[0]
    return ts, np.array(rows, dtype=np.float32)


def _field(rec, name):
    """Read a projectaria_tools record field that may be an attribute OR a method.

    The obvious `getattr(rec, name, None)` then `if x is None: x = rec.name()` is WRONG:
    when the binding exposes a method, getattr returns the bound method object, which is
    not None, so the fallback never fires and float() raises on a method. Test callable()
    instead.
    """
    v = getattr(rec, name, None)
    try:
        return v() if callable(v) else v
    except Exception:
        return None


def measure_imu_rate(imu_ts):
    """Actual IMU sampling rate from timestamps -- do not assume 1 kHz."""
    if imu_ts is None or len(imu_ts) < 2:
        return dict(n_samples=0, implied_hz=float("nan"), span_s=0.0)
    dts = np.diff(np.asarray(imu_ts, dtype=np.float64)) / 1e6
    dts = dts[dts > 0]
    med = float(np.median(dts)) if len(dts) else float("nan")
    return dict(n_samples=int(len(imu_ts)),
                implied_hz=(1.0 / med) if med > 0 else float("nan"),
                span_s=float((imu_ts[-1] - imu_ts[0]) / 1e6))


def window_imu_samples(imu_ts, imu_arr, t0_us, t1_us):
    """ALL IMU samples in [t0, t1). Returns dict(rows=[[6 channels],...], n)."""
    if imu_ts is None or imu_arr is None:
        return dict(rows=[], n=0)
    m = (imu_ts >= t0_us) & (imu_ts < t1_us)
    rows = imu_arr[m]
    return dict(rows=[[float(v) for v in r] for r in rows], n=int(len(rows)))


def window_imu_binned(imu_ts, imu_arr, t0_us, t1_us, n_bins):
    """IMU in [t0, t1), averaged within `n_bins` equal sub-intervals.

    Why bin rather than store every sample: Aria's IMU runs at ~800-1000 Hz, so a 1 s
    window holds ~1000 rows x 6 channels. Written out in full that is ~50 KB per CSV row
    -- around 20 MB for a 2-video build and over a GB across all 143 sequences, which
    makes the CSV impractical to load. `n_bins=10` matches the ~10 gaze samples per
    window, so the IMU column lines up with gaze_xy_window row for row.

    Averaging within each bin is an anti-aliased downsample. Keeping every Nth sample
    instead would fold the high-frequency vibration content straight into the signal at
    the wrong frequency, which is exactly the artefact that would look like head motion.

    Note the averaging is over the raw VECTORS, so a bin containing equal-and-opposite
    rotation averages towards zero. That is correct for a mean angular velocity but hides
    how much the head actually moved -- `imu_gyro_mag_mean` in the caller is computed on
    the unbinned samples for that reason.

    Returns dict(rows=(n_bins, 6) float list, n_raw, n_empty). Bins that caught no sample
    are NaN, not zero: zero is a physically meaningful accelerometer reading and must not
    be confused with missing data.
    """
    empty = dict(rows=[], n_raw=0, n_empty=int(n_bins))
    if imu_ts is None or imu_arr is None or n_bins <= 0:
        return empty
    m = (imu_ts >= t0_us) & (imu_ts < t1_us)
    ts, a = imu_ts[m], imu_arr[m]
    if len(ts) == 0:
        return empty

    edges = np.linspace(float(t0_us), float(t1_us), int(n_bins) + 1)
    # np.digitize with the interior edges gives the bin index directly
    idx = np.clip(np.digitize(ts, edges[1:-1]), 0, int(n_bins) - 1)

    out = np.full((int(n_bins), a.shape[1]), np.nan, dtype=np.float64)
    n_empty = 0
    for b in range(int(n_bins)):
        sel = idx == b
        if sel.any():
            out[b] = a[sel].mean(axis=0)
        else:
            n_empty += 1
    return dict(rows=[[float(v) for v in r] for r in out],
                n_raw=int(len(ts)), n_empty=int(n_empty))


def subsample_frames(mp4, out_dir, target_fps=1.0, max_frames=None):
    """Decode to `target_fps`, writing one JPEG per kept frame.

    `max_frames` caps how many SOURCE frames are read (not kept), so
    max_frames = seconds * native_fps bounds the wall-clock cost on long videos.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(mp4); native = cap.get(cv2.CAP_PROP_FPS) or 20.0
    step = max(1, int(round(native / target_fps)))
    paths, ts_us, kept, i = [], [], 0, 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames is not None and i >= max_frames):
            break
        if i % step == 0:
            p = os.path.join(out_dir, f"frame_{kept:05d}.jpg")
            cv2.imwrite(p, fr)                      # save BGR directly
            paths.append(p); ts_us.append((i / native) * 1e6); kept += 1
        i += 1
    cap.release()
    return paths, np.array(ts_us), native