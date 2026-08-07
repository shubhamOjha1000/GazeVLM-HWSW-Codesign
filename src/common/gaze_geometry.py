import numpy as np

def yawpitch_to_unit_vec(yaw, pitch):
    """Gaze direction as a 3D unit vector on the sphere."""
    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)
    v = np.array([x, y, z], dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-8)

def yawpitch_to_norm_xy(yaw, pitch, fov_x, fov_y, yaw_sign=1, pitch_sign=-1):
    """Pinhole (prototype-grade) projection of gaze angle to normalized image (x, y)."""
    x = 0.5 + yaw_sign * np.tan(yaw) / (2 * np.tan(fov_x / 2))
    y = 0.5 + pitch_sign * np.tan(pitch) / (2 * np.tan(fov_y / 2))
    return float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))

def measure_gaze_rate(g_ts_us):
    """Verify the ACTUAL gaze sampling rate from timestamps (do not assume 10 Hz)."""
    ts = np.asarray(g_ts_us, dtype=np.float64)
    dts = np.diff(ts) / 1e6
    dts = dts[dts > 0]
    median_dt = float(np.median(dts)) if len(dts) else float("nan")
    return dict(
        n_samples=int(len(ts)),
        median_dt_ms=median_dt * 1000,
        implied_hz=(1.0 / median_dt) if median_dt > 0 else float("nan"),
        min_dt_ms=float(dts.min() * 1000) if len(dts) else float("nan"),
        max_dt_ms=float(dts.max() * 1000) if len(dts) else float("nan"),
        span_s=float((ts[-1] - ts[0]) / 1e6) if len(ts) else 0.0,
    )

def window_gaze_rates(g_ts_us, g_yaw, g_pit, t0_us, t1_us):
    """Per-step eye angular velocity in [t0, t1), as an (n-1, 3) signal.

    n gaze samples in the window give n-1 consecutive steps. Each step carries three
    channels, all in rad/s, computed as discrete derivatives on the GAZE clock:

        omega_yaw(t)   = [theta(t) - theta(t - dt)] / dt     signed, + = rightward
        omega_pitch(t) = [phi(t)   - phi(t - dt)]   / dt     signed, + = upward
        omega_mag(t)   = sqrt(omega_yaw^2 + omega_pitch^2)   always >= 0

    Returned as `rates`, shape (n-1, 3) with columns [yaw_rate, pitch_rate, magnitude] --
    the 3-channel form the gating CNN consumes.

    NOTE: yaw and pitch are not orthogonal coordinates on a sphere, so `omega_mag` is not
    exactly the true angular speed: the yaw term is overstated by 1/cos(pitch), which is
    ~33% at the -41 deg pitch this dataset reaches. Multiply the yaw rate by cos(pitch)
    if a sphere-exact magnitude is wanted.
    """
    m = (g_ts_us >= t0_us) & (g_ts_us < t1_us)
    ts, ya, pi = g_ts_us[m], g_yaw[m], g_pit[m]
    good = np.isfinite(ya) & np.isfinite(pi)
    ts, ya, pi = ts[good], ya[good], pi[good]
    if len(ts) < 2:
        return dict(rates=np.zeros((0, 3), dtype=np.float64),
                    n_samples=int(len(ts)), n_velocities=0)
    dts = np.diff(ts) / 1e6
    dts = np.where(dts <= 0, 1e-6, dts)              # n-1 gaze-clock intervals (~0.1 s)
    w_yaw = np.diff(ya) / dts                        # signal 1
    w_pit = np.diff(pi) / dts                        # signal 2
    w_mag = np.hypot(w_yaw, w_pit)                   # signal 3
    return dict(
        rates=np.stack([w_yaw, w_pit, w_mag], axis=1),   # (n-1, 3)
        n_samples=int(len(ts)), n_velocities=int(len(dts)),
    )

def window_gaze_samples(g_ts_us, g_yaw, g_pit, t0_us, t1_us, project=None):
    """All valid raw gaze samples in [t0, t1): per-sample yaw, pitch, 3D unit vector,
    projected (x, y) and the out-of-FOV flag. Populates the granular per-window columns
    (variable length).

    The projection itself lives in `project` (see make_gaze_projector), so the CPF->camera
    transform and the native->upright rotation are applied here automatically -- there is
    only one implementation. What this function must NOT do is discard the out-of-FOV flag
    that the calibration projector returns alongside (x, y): when gaze falls outside the
    lens, (0.5, 0.5) is substituted, and without the flag those invented centre points are
    indistinguishable from a genuine centre gaze in the training input.
    """
    m = (g_ts_us >= t0_us) & (g_ts_us < t1_us)
    ya, pi = g_yaw[m], g_pit[m]
    good = np.isfinite(ya) & np.isfinite(pi)
    ya, pi = ya[good], pi[good]
    yaws, pitches, vecs, xs, ys, oofs = [], [], [], [], [], []
    for y, p in zip(ya, pi):
        v = yawpitch_to_unit_vec(y, p)
        yaws.append(float(y)); pitches.append(float(p))
        vecs.append((float(v[0]), float(v[1]), float(v[2])))
        if project is not None:
            out = project(y, p)                              # calib: (x, y, oof); pinhole: (x, y)
            xs.append(float(out[0])); ys.append(float(out[1]))
            oofs.append(bool(out[2]) if len(out) > 2 else False)
        else:
            # keep every list the same length so per-sample indices stay aligned
            xs.append(float("nan")); ys.append(float("nan")); oofs.append(True)
    return dict(yaw=yaws, pitch=pitches, vec3d=vecs, x=xs, y=ys, oof=oofs,
                n=len(yaws), n_oof=int(sum(oofs)))

def native_to_upright_cw90(x, y):
    """Native Aria RGB sensor coords -> upright-image coords.

    The Aria RGB sensor stores frames rotated; the preview MP4 this pipeline decodes is
    already upright, so projected pixels (which land in the NATIVE frame, exactly as in
    Meta's own tutorials) must be rotated to match. This is the coordinate equivalent of
    `np.rot90(img, k=3)` (90 deg clockwise), the rotation the projectaria_tools image
    utilities prescribe. Square frames make the mismatch invisible to any size or
    aspect-ratio check -- only an axis-sweep or a visual overlay catches it.
    """
    return 1.0 - y, x


def make_gaze_projector(projection="pinhole", vrs_path=None,
                        fov_x=None, fov_y=None, yaw_sign=1, pitch_sign=-1,
                        stream_label="camera-rgb", depth_m=1.0, rotate_cw90=True):
    """Return a fn(yaw, pitch) -> (x_norm, y_norm[, oof]) selected at runtime.

    rotate_cw90: map native sensor coords to the upright preview-MP4 frame. Set False
    only when projecting onto raw VRS images, which are in the native orientation.
    """
    if projection == "pinhole":
        def _proj(yaw, pitch):
            return yawpitch_to_norm_xy(yaw, pitch, fov_x, fov_y, yaw_sign, pitch_sign)
        return _proj
    if projection == "calibration":
        # Transform chain copied from Meta's own AEA quickstart tutorial
        # (`draw_eye_gaze`), which is the reference implementation:
        #     gaze_cpf = get_eyegaze_point_at_depth(yaw, pitch, depth_m)
        #     gaze_cam = T_device_camera.inverse() @ T_device_cpf @ gaze_cpf
        #     px       = camera_calib.project(gaze_cam)
        # CPF (Central Pupil Frame) is anchored between the eyes, ~6 cm from the RGB
        # camera and rotated relative to it; project() expects a point already in the
        # camera's own frame. Feeding it a CPF point offsets every gaze dot by that
        # baseline. Note this uses the factory-calibrated device<->cpf and
        # device<->camera extrinsics rather than the get_transform_cpf_sensor()
        # shortcut, which can return nominal CAD values instead.
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.mps import get_eyegaze_point_at_depth
        provider = data_provider.create_vrs_data_provider(vrs_path)
        device_calib = provider.get_device_calibration()
        cam_calib = device_calib.get_camera_calib(stream_label)
        T_device_cpf = device_calib.get_transform_device_cpf()
        T_device_cam = cam_calib.get_transform_device_camera()
        W, H = cam_calib.get_image_size()
        def _proj(yaw, pitch, depth=depth_m):
            gaze_cpf = get_eyegaze_point_at_depth(yaw, pitch, depth)
            gaze_cam = T_device_cam.inverse() @ T_device_cpf @ gaze_cpf
            px = cam_calib.project(gaze_cam)
            if px is None:
                return 0.5, 0.5, True                          # out-of-FOV fallback (flagged)
            u, v = np.asarray(px, dtype=np.float64).flatten()[:2]
            x, y = u / W, v / H                                # native sensor orientation
            if rotate_cw90:
                x, y = native_to_upright_cw90(x, y)
            return float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1)), False
        return _proj
    raise ValueError(f"unknown projection: {projection}")


def official_reprojection(vrs_path, gaze_csv_path, stream_label="camera-rgb",
                          depth_m=1.0, rotate_cw90=True):
    """Oracle: reproject gaze with Meta's own `get_gaze_vector_reprojection` helper.

    Returns (xy_norm[N,2], oof_mask[N]) for every sample in the MPS eye-gaze file, in
    file order. Not used by the pipeline -- this is the reference to diff
    `make_gaze_projector("calibration", ...)` against, so the projection can be checked
    numerically instead of by eye. A correct projector agrees with this to float noise.

    `rotate_cw90` must match the projector under test, or the comparison is meaningless.
    """
    from projectaria_tools.core import data_provider, mps
    from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection

    provider = data_provider.create_vrs_data_provider(vrs_path)
    device_calib = provider.get_device_calibration()
    cam_calib = device_calib.get_camera_calib(stream_label)
    W, H = cam_calib.get_image_size()

    xy, oof = [], []
    for g in mps.read_eyegaze(gaze_csv_path):
        px = get_gaze_vector_reprojection(g, stream_label, device_calib, cam_calib, depth_m)
        if px is None:
            xy.append((0.5, 0.5)); oof.append(True)
            continue
        u, v = np.asarray(px, dtype=np.float64).flatten()[:2]
        x, y = u / W, v / H
        if rotate_cw90:
            x, y = native_to_upright_cw90(x, y)
        xy.append((float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))))
        oof.append(False)
    return np.array(xy), np.array(oof)