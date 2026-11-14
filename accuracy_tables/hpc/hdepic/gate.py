"""Runbook section 17: the gate on HD-EPIC's gaze questions. CPU only, base environment.

    python -u -m accuracy_tables.hpc.hdepic.gate resample-check     step 1: D4's resampler gives the existing 30 Hz rows exactly
    python -u -m accuracy_tables.hpc.hdepic.gate imu                step 1: HD-EPIC's right IMU against AEA's: rate, units, axes
    python -u -m accuracy_tables.hpc.hdepic.gate rows               step 2: one rows CSV per question
    python -u -m accuracy_tables.hpc.hdepic.gate score              step 2: the 5 fold models averaged, on every row
    python -u -m accuracy_tables.hpc.hdepic.gate stats              step 2: the gate's statistics against AEA and Nymeria, and the VOR check

The rows are section 9's (accuracy_tables.hpc.newvideos.build_rows.pair_rows, unchanged), with two settings HD-EPIC needs:
  windows   each candidate's window is the 1 s before it (pair_rows' window_us), the window length the gate trained on at 1 fps. The
            candidates here are 0.03 to 0.16 s apart, so neighbouring windows overlap; each still has 9 gaze steps and 10 IMU bins.
  gaze rate D4: one resampler for any rate within 1 Hz of a whole multiple of 10 Hz, every (rate / 10)-th sample with no averaging, as
            450298f does at 30 Hz: step 1 at 10 Hz, 6 at 60 Hz. Any other rate stops. resample-check proves it gives section 9's rows
            byte for byte on every Nymeria 30 Hz video.
Each question is its own clip (its pseudo-video name, as in plan.json), and every stream stays in device time.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from accuracy_tables.hpc.hdepic import data
from accuracy_tables.hpc.newvideos import build_rows, check_gate, paths

WINDOW_US = 1_000_000
GATE_HZ = 10.0
VOR_CLEARLY_NEGATIVE = -0.5            # AEA and Nymeria: about -0.88; anything above this means the IMU axes or sign are wrong


# ---------------------------------------------------------------------------- D4: the gaze rate
def gate_step_any(t_us, target=GATE_HZ, tolerance=1.0):
    """(rate in Hz, every how many samples reach 10 Hz) for any rate within tolerance of a whole multiple of 10 Hz; stops otherwise."""
    hz = 1e6 / float(np.median(np.diff(np.asarray(t_us, dtype=np.float64))))
    step = int(round(hz / target))
    if step < 1 or abs(hz / step - target) > tolerance:
        raise ValueError(f"gaze at {hz:.2f} Hz: no whole step gives {target:g} Hz")
    return hz, step


def resample_check(videos=None, out_dir=None):
    """Section 9's rows of each Nymeria video at 30 Hz, built again with gate_step_any's step into out_dir, against the existing file
    (only read) -> (videos compared, videos identical, [videos that differ])."""
    videos = videos if videos is not None else paths.read_list("nymeria")
    out_dir = out_dir or os.path.join(data.derived_root(), "resample_check")
    os.makedirs(out_dir, exist_ok=True)
    compared, differ = 0, []
    for v in videos:
        stats = paths.rows_stats(v)
        if not os.path.exists(stats):
            continue
        with open(stats) as f:
            if json.load(f).get("gate_gaze_step", 1) != 3:
                continue
        with np.load(paths.gaze_npz(v)) as z:
            g = {k: z[k] for k in ("t_us", "yaw", "pitch", "x", "y", "oof")}
        old_hz, old_step = build_rows.gate_gaze_step(g["t_us"])
        hz, step = gate_step_any(g["t_us"])
        meta = paths.frame_meta(v)
        f_ts = np.asarray(meta["ts"], dtype=np.float64) * 1e6
        imu_ts, imu_arr = build_rows.load_imu(os.path.join(paths.video_dir(v), "imu.npz"))
        project = build_rows.lookup_projector(g["yaw"], g["pitch"], g["x"], g["y"], g["oof"])
        rows = build_rows.pair_rows(v, f_ts, g["t_us"], g["yaw"], g["pitch"], project, imu_ts, imu_arr, gate_step=step)
        path = os.path.join(out_dir, f"{v}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        with open(path, "rb") as a, open(paths.rows_csv(v), "rb") as b:
            same = a.read() == b.read() and step == old_step
        compared += 1
        if not same:
            differ.append(v)
        print(f"   {v}: {hz:.2f} Hz, step {step} (section 9: {old_step}): {'identical' if same else 'DIFFERENT'}", flush=True)
    return compared, compared - len(differ), differ


# ---------------------------------------------------------------------------- the IMU against AEA's
def imu_profile(path):
    """The right IMU of one imu.npz: rate, mean accel vector and its magnitude, gyro magnitude, per-axis gyro spread."""
    with np.load(path) as z:
        t, a, g = z["right_t_ns"], z["right_accel"].astype(np.float64), z["right_gyro"].astype(np.float64)
    mean_a = a.mean(0)
    return dict(hz=data.rate_hz(t, 1e9), accel_mag=float(np.linalg.norm(a, axis=1).mean()), accel_dir=mean_a / np.linalg.norm(mean_a),
                gyro_mag_p50=float(np.median(np.linalg.norm(g, axis=1))), gyro_std=g.std(0))


def imu_check(hd_videos, aea_videos):
    """HD-EPIC's IMU against AEA's -> (table, problems). The gate reads gyro columns 3 to 5 of [accel xyz, gyro xyz] from the right
    IMU (build_rows.load_imu), so the right IMU, m/s^2 and rad/s, and the same device axes must hold here."""
    rows = []
    for name, videos, folder in (("hdepic", hd_videos, data.video_dir), ("aea", aea_videos, paths.video_dir)):
        for v in videos:
            p = os.path.join(folder(v), "imu.npz")
            if os.path.exists(p):
                rows.append(dict(source=name, video=v, **imu_profile(p)))
    t = pd.DataFrame(rows)
    found = []
    hd, aea = t[t.source == "hdepic"], t[t.source == "aea"]
    if not len(hd) or not len(aea):
        return t, ["no HD-EPIC or no AEA imu.npz to compare"]
    ref = np.mean(np.stack(aea.accel_dir), 0)
    ref /= np.linalg.norm(ref)
    t["accel_dir_cos_to_aea"] = [float(np.dot(d, ref)) for d in t.accel_dir]
    hdr = t[t.source == "hdepic"]
    if not (8.8 <= hdr.accel_mag.median() <= 10.8):
        found.append(f"accel magnitude {hdr.accel_mag.median():.2f}: not m/s^2")
    if abs(hdr.hz.median() / aea.hz.median() - 1) > 0.1:
        found.append(f"right IMU at {hdr.hz.median():.0f} Hz, AEA's at {aea.hz.median():.0f} Hz")
    if hdr.accel_dir_cos_to_aea.median() < 0.9:
        found.append(f"gravity's direction in the device frame is {hdr.accel_dir_cos_to_aea.median():.2f} cos from AEA's: other axes")
    if hdr.gyro_mag_p50.median() > 2.0:
        found.append(f"gyro magnitude {hdr.gyro_mag_p50.median():.2f}: not rad/s")
    return t, found


# ---------------------------------------------------------------------------- rows and scores
def load_gaze(video):
    with np.load(data.gaze_npz(video)) as z:
        return {k: z[k] for k in ("t_us", "yaw", "pitch", "x", "y", "oof")}


def load_imu_device(path):
    """The right IMU as build_rows.load_imu gives it ([accel xyz, gyro xyz]), in device microseconds instead of rebased."""
    with np.load(path) as z:
        t_ns, accel, gyro = z["right_t_ns"], z["right_accel"], z["right_gyro"]
    return t_ns.astype(np.float64) / 1e3, np.concatenate([accel, gyro], axis=1).astype(np.float32)


def question_rows(q, g, imu_ts, imu_arr):
    """One question's gate rows: its candidates in device time, each with the 1 s window before it."""
    _, step = gate_step_any(g["t_us"])
    project = build_rows.lookup_projector(g["yaw"], g["pitch"], g["x"], g["y"], g["oof"])
    return build_rows.pair_rows(q["pseudo"], np.asarray(q["device_us"], dtype=np.float64), g["t_us"].astype(np.float64), g["yaw"],
                                g["pitch"], project, imu_ts, imu_arr, gate_step=step, window_us=WINDOW_US)


def rows_cmd(a):
    plan = data.read_plan()
    by_video = {}
    for q in plan["questions"]:
        by_video.setdefault(q["video"], []).append(q)
    stats = []
    for i, (v, qs) in enumerate(sorted(by_video.items())):
        g = load_gaze(v)
        hz, step = gate_step_any(g["t_us"])
        imu_ts, imu_arr = load_imu_device(os.path.join(data.video_dir(v), "imu.npz"))
        for q in qs:
            rows = pd.DataFrame(question_rows(q, g, imu_ts, imu_arr))
            out = data.rows_csv(q["pseudo"])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            rows.to_csv(out + ".tmp", index=False)
            os.replace(out + ".tmp", out)
            stats.append(dict(pseudo=q["pseudo"], type=q["type"], video=v, gaze_hz=hz, step=step, rows=len(rows),
                              distinct_gaze_inputs=int(rows.gaze_rates_window.nunique()),
                              distinct_inputs=int((rows.gaze_rates_window.astype(str) + "|" + rows.imu_window.astype(str)).nunique()),
                              velocities_min=int(rows.n_velocities.min()), imu_min=int(rows.n_imu_in_gap.min())))
        print(f"   {i + 1} of {len(by_video)} videos: {v}, gaze {hz:.2f} Hz, step {step}, {len(qs)} questions", flush=True)
    s = pd.DataFrame(stats)
    s.to_csv(os.path.join(data.derived_root(), "rows_stats.csv"), index=False)
    for qtype, x in s.groupby("type", sort=False):
        print(f"{qtype}: {len(x)} questions, rows per question median {x.rows.median():g}; distinct 1 s gaze inputs per question median "
              f"{x.distinct_gaze_inputs.median():g} (min {x.distinct_gaze_inputs.min()}); distinct gaze + IMU inputs median "
              f"{x.distinct_inputs.median():g}; fewest gaze velocities in a window {x.velocities_min.min()}, fewest IMU samples "
              f"{x.imu_min.min()}", flush=True)
    print(f"ROWS WRITTEN: {len(s)} questions", flush=True)


def score_cmd(a):
    from accuracy_tables.hpc.newvideos import score_gate
    plan = data.read_plan()
    d = data.folds_dir()
    os.makedirs(d, exist_ok=True)
    missing = [q["pseudo"] for q in plan["questions"] if not os.path.exists(data.rows_csv(q["pseudo"]))]
    if missing:
        sys.exit(f"{len(missing)} questions have no rows, for example {missing[:5]}. Rerun the rows step.")
    rows = pd.concat([pd.read_csv(data.rows_csv(q["pseudo"])) for q in plan["questions"]], ignore_index=True)
    rows.insert(0, "idx", range(len(rows)))
    test = os.path.join(d, "test.csv")
    rows.to_csv(test, index=False)
    rows.head(0).to_csv(os.path.join(d, "train_val_fold1.csv"), index=False)
    out = score_gate.score_ensemble(test, score_gate.load_models(paths.ckpt_dir()))
    out.to_csv(data.preds_csv() + ".tmp", index=False)
    os.replace(data.preds_csv() + ".tmp", data.preds_csv())
    changes = out.groupby("sequence", sort=False)["pred"].apply(lambda p: int((np.diff(p.to_numpy()) != 0).sum()) + 1)
    types = {q["pseudo"]: q["type"] for q in plan["questions"]}
    for qtype in data.GAZE_TYPES:
        c = changes[[s for s in changes.index if types.get(s) == qtype]]
        print(f"{qtype}: distinct gate decisions per question (runs of one label) median {c.median():g}, min {c.min()}, max {c.max()}",
              flush=True)
    print(f"SCORED: {len(out)} rows of {out.sequence.nunique()} questions -> {data.preds_csv()}", flush=True)


def hdepic_rows():
    cols = ["sequence", "feat_frame_2", "n_gaze_in_gap", "n_oof_in_gap", "n_imu_in_gap", check_gate.EYE_COL, check_gate.IMU_COL]
    rows = pd.read_csv(os.path.join(data.folds_dir(), "test.csv"), usecols=cols)
    preds = pd.read_csv(data.preds_csv(), usecols=["feat_frame_2", "pred"])
    out = rows.merge(preds, on="feat_frame_2", how="left", validate="one_to_one")
    if out["pred"].isna().any():
        raise SystemExit(f"{int(out['pred'].isna().sum())} rows have no prediction. Rerun the score step.")
    out["frame"] = out["feat_frame_2"].map(check_gate.pol.frame_no)
    return out.sort_values(["sequence", "frame"]).reset_index(drop=True)


def stats_cmd(a):
    from eyehead.infer import load_ckpt
    plan = data.read_plan()
    span = {q["pseudo"]: q["mp4_s"][-1] - q["mp4_s"][0] for q in plan["questions"]}
    types = {q["pseudo"]: q["type"] for q in plan["questions"]}
    ck = load_ckpt(os.path.join(paths.ckpt_dir(), "fold1.pt"), "cpu")
    df = hdepic_rows()
    table = {}
    for name, keep in (("hdepic", None), *((t, t) for t in data.GAZE_TYPES)):
        part = df if keep is None else df[df.sequence.map(types) == keep]
        per = pd.DataFrame([check_gate.video_summary(v, x["pred"].to_numpy().astype(int), x["n_gaze_in_gap"].to_numpy(),
                                                     x["n_oof_in_gap"].to_numpy(), x["n_imu_in_gap"].to_numpy())
                            for v, x in part.groupby("sequence", sort=False)])
        s = dict(check_gate.group_summary(per, part["pred"].to_numpy().astype(int)), **check_gate.input_stats(part, ck))
        seconds = float(sum(span[v] for v in per.video))
        s["marks_per_min"] = 60 * per["marks"].sum() / seconds            # real seconds: candidates aren't 1 s apart here
        s["seconds_p50"] = float(np.median([span[v] for v in per.video]))
        s["seconds_max"] = float(max(span[v] for v in per.video))
        table[name] = s
    for ref in ("aea12", "nymeria"):
        try:
            rdf = check_gate.load_group(ref)
        except Exception as ex:                                        # a reference that isn't there is reported, not fatal
            print(f"   {ref}: not compared ({type(ex).__name__}: {str(ex)[:120]})", flush=True)
            continue
        per = pd.DataFrame([check_gate.video_summary(v, x["pred"].to_numpy().astype(int), x["n_gaze_in_gap"].to_numpy(),
                                                     x["n_oof_in_gap"].to_numpy(), x["n_imu_in_gap"].to_numpy())
                            for v, x in rdf.groupby("sequence", sort=False)])
        table[ref] = dict(check_gate.group_summary(per, rdf["pred"].to_numpy().astype(int)), **check_gate.input_stats(rdf, ck))
    t = pd.DataFrame(table)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(), flush=True)
    t.to_csv(os.path.join(data.derived_root(), "gate_stats.csv"))
    vor = table["hdepic"]["vor_median_cos"]
    if not np.isfinite(vor) or vor > VOR_CLEARLY_NEGATIVE:
        sys.exit(f"VOR CHECK FAILED: HD-EPIC's vor_median_cos is {vor:.3f}, not clearly negative (AEA and Nymeria about -0.88). The IMU "
                 "axes or sign are wrong. Stop and tell the user.")
    print(f"VOR CHECK OK: vor_median_cos {vor:.3f}. STOP: show the user this table and wait for their OK.", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def resample(a):
        n, same, differ = resample_check()
        if not n:
            sys.exit("no Nymeria video at 30 Hz found to compare. Stop and tell the user.")
        if differ:
            sys.exit(f"REBUILD DIFFERS on {len(differ)} of {n} Nymeria 30 Hz videos: {differ[:5]}. Stop and tell the user.")
        print(f"REBUILD MATCHES: {same} of {n} Nymeria 30 Hz videos, byte for byte, with the general resampler", flush=True)

    def imu(a):
        plan = data.read_plan()
        t, found = imu_check(plan["videos"], paths.read_list("aea12"))
        cols = ["source", "video", "hz", "accel_mag", "accel_dir_cos_to_aea", "gyro_mag_p50"]
        print(t[[c for c in cols if c in t]].round(3).to_string(index=False), flush=True)
        if found:
            sys.exit("IMU DIFFERS FROM AEA: " + "; ".join(found) + ". Stop and tell the user.")
        print("IMU MATCHES AEA: right IMU, m/s^2 and rad/s, the same device axes; the gate reads gyro columns 3 to 5", flush=True)

    for name, fn in (("resample-check", resample), ("imu", imu), ("rows", rows_cmd), ("score", score_cmd), ("stats", stats_cmd)):
        sub.add_parser(name).set_defaults(run=fn)
    a = ap.parse_args(argv)
    a.run(a)


if __name__ == "__main__":
    main()
