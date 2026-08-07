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
        # rec exposes capture timestamp (ns) + accel/gyro xyz; names vary by version
        t_ns = getattr(rec, "capture_timestamp_ns", None)
        if t_ns is None:
            t_ns = rec.capture_timestamp_ns()  # some versions expose as a method
        acc = getattr(rec, "accel_msec2", None);  acc = acc() if callable(acc) else acc
        gyr = getattr(rec, "gyro_radsec", None);   gyr = gyr() if callable(gyr) else gyr
        ts.append(float(t_ns) / 1e3)  # ns -> us
        rows.append([float(acc[0]), float(acc[1]), float(acc[2]),
                     float(gyr[0]), float(gyr[1]), float(gyr[2])])
    if not ts:
        return None, None
    ts = np.array(ts); ts = ts - ts[0]
    return ts, np.array(rows, dtype=np.float32)


def window_imu_samples(imu_ts, imu_arr, t0_us, t1_us):
    """ALL IMU samples in [t0, t1). Returns dict(rows=[[6 channels],...], n)."""
    if imu_ts is None or imu_arr is None:
        return dict(rows=[], n=0)
    m = (imu_ts >= t0_us) & (imu_ts < t1_us)
    rows = imu_arr[m]
    return dict(rows=[[float(v) for v in r] for r in rows], n=int(len(rows)))


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