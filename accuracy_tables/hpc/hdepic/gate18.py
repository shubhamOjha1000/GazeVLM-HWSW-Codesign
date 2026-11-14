"""Runbook section 18: the gate on HD-EPIC's 6 groups. CPU only, base environment. Section 17's gate code (gate.py), on section 18's plan.

    python -u -m accuracy_tables.hpc.hdepic.gate18 imu              step 1: every planned video's right IMU against AEA's
    python -u -m accuracy_tables.hpc.hdepic.gate18 rows             step 2: one rows CSV per question
    python -u -m accuracy_tables.hpc.hdepic.gate18 score            step 2: the 5 fold models averaged, on every row
    python -u -m accuracy_tables.hpc.hdepic.gate18 stats            step 2: the statistics per group against AEA and Nymeria, the VOR check
    python -u -m accuracy_tables.hpc.hdepic.gate18 compare          step 2: what D2's query-frame exception does to ours' frames

The rows are section 9's pair_rows, unchanged, with section 17's settings: each candidate's window is the 1 s before it (window_us) and
any gaze rate is taken to 10 Hz by section 17's resampler (D4). A question's rows are built segment by segment, each from its own video's
gaze and IMU, and chained into one clip (one row per candidate after the question's first), so every question is one clip as in section 17.
The row at a segment's first candidate is computed and never read: the budget rule treats that frame as a frame 0 of its own (the plan's
`starts`), so the gate runs per segment and never across a gap. Every stream stays in device time.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from accuracy_tables.hpc.hdepic import data, gate, s18
from accuracy_tables.hpc.newvideos import build_rows, check_gate, paths

WINDOW_US = s18.WINDOW_US


def read_plan():
    with open(s18.plan_json()) as f:
        return json.load(f)


class Streams:
    """Each real video's gaze, gate step, projector and IMU, loaded once and kept for the last few videos."""

    def __init__(self, keep=4):
        self.keep, self.cache = keep, {}

    def __call__(self, v):
        if v not in self.cache:
            if len(self.cache) >= self.keep:
                self.cache.pop(next(iter(self.cache)))
            g = gate.load_gaze(v)
            hz, step = gate.gate_step_any(g["t_us"])
            imu_ts, imu_arr = gate.load_imu_device(os.path.join(data.video_dir(v), "imu.npz"))
            project = build_rows.lookup_projector(g["yaw"], g["pitch"], g["x"], g["y"], g["oof"])
            self.cache[v] = dict(g=g, hz=hz, step=step, imu_ts=imu_ts, imu_arr=imu_arr, project=project)
        return self.cache[v]


def question_rows(q, streams):
    """One question's gate rows, segment by segment, chained: a row per candidate after the first, named by its place in the question."""
    rows = []
    bounds = [0] + list(q["starts"]) + [q["K"]]
    for s0, s1 in zip(bounds, bounds[1:]):
        st = streams(q["videos"][q["video_of"][s0]])
        t = np.asarray(q["device_us"][s0:s1], dtype=np.float64)
        pos = list(range(s0, s1))
        if s0 > 0:                                          # a row for the segment's first candidate too (computed, not read)
            t, pos = np.concatenate([[t[0] - WINDOW_US], t]), [s0 - 1] + pos
        g = st["g"]
        part = build_rows.pair_rows(q["pseudo"], t, g["t_us"].astype(np.float64), g["yaw"], g["pitch"], st["project"], st["imu_ts"],
                                    st["imu_arr"], gate_step=st["step"], window_us=WINDOW_US)
        for r, i in zip(part, range(1, len(pos))):
            r["feat_frame_1"] = build_rows.feature_name(q["pseudo"], pos[i - 1])
            r["feat_frame_2"] = build_rows.feature_name(q["pseudo"], pos[i])
        rows += part
    return rows


def rows_stats(q, rows):
    return dict(pseudo=q["pseudo"], group=q["group"], type=q["type"], rows=len(rows), segments=q["n_segments"],
                distinct_gaze_inputs=int(rows.gaze_rates_window.nunique()), velocities_min=int(rows.n_velocities.min()),
                imu_min=int(rows.n_imu_in_gap.min()), no_gaze_windows=int((rows.n_gaze_in_gap == 0).sum()))


def rows_cmd(a):
    plan = read_plan()
    streams = Streams()
    stats = []
    for i, q in enumerate(sorted(plan["questions"], key=lambda q: (q["videos"][0], q["qid"]))):
        rows = pd.DataFrame(question_rows(q, streams))
        if len(rows) != q["K"] - 1:
            sys.exit(f"{q['pseudo']}: {len(rows)} rows for {q['K']} candidates. Stop and tell the user.")
        out = s18.rows_csv(q["pseudo"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        rows.to_csv(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
        stats.append(rows_stats(q, rows))
        if i % 500 == 0:
            print(f"   {i} of {len(plan['questions'])} questions", flush=True)
    for v, st in sorted(streams.cache.items()):
        print(f"   {v}: gaze {st['hz']:.2f} Hz, step {st['step']}", flush=True)
    s = pd.DataFrame(stats)
    s.to_csv(os.path.join(s18.root(), "rows_stats.csv"), index=False)
    for g, x in s.groupby("group"):
        print(f"{g}: {len(x)} questions, rows per question median {x.rows.median():g}; distinct 1 s gaze inputs median "
              f"{x.distinct_gaze_inputs.median():g}; fewest gaze velocities in a window {x.velocities_min.min()}, fewest IMU samples "
              f"{x.imu_min.min()}; windows with no gaze {int(x.no_gaze_windows.sum())} of {int(x.rows.sum())}", flush=True)
    print(f"ROWS WRITTEN: {len(s)} questions", flush=True)


def score_rows(questions, folds, preds_path, rows_of):
    """Every question's rows into <folds>/test.csv, the 5 fold models averaged on them -> preds_path."""
    from accuracy_tables.hpc.newvideos import score_gate
    os.makedirs(folds, exist_ok=True)
    rows = pd.concat([rows_of(q) for q in questions], ignore_index=True)
    rows.insert(0, "idx", range(len(rows)))
    test = os.path.join(folds, "test.csv")
    rows.to_csv(test, index=False)
    rows.head(0).to_csv(os.path.join(folds, "train_val_fold1.csv"), index=False)
    out = score_gate.score_ensemble(test, score_gate.load_models(paths.ckpt_dir()))
    out.to_csv(preds_path + ".tmp", index=False)
    os.replace(preds_path + ".tmp", preds_path)
    return out


def score_cmd(a):
    plan = read_plan()
    missing = [q["pseudo"] for q in plan["questions"] if not os.path.exists(s18.rows_csv(q["pseudo"]))]
    if missing:
        sys.exit(f"{len(missing)} questions have no rows, for example {missing[:5]}. Rerun the rows step.")
    out = score_rows(plan["questions"], s18.folds_dir(), s18.preds_csv(), lambda q: pd.read_csv(s18.rows_csv(q["pseudo"])))
    changes = out.groupby("sequence", sort=False)["pred"].apply(lambda p: int((np.diff(p.to_numpy()) != 0).sum()) + 1)
    group = {q["pseudo"]: q["group"] for q in plan["questions"]}
    for g in sorted(set(group.values())):
        c = changes[[s for s in changes.index if group.get(s) == g]]
        print(f"{g}: distinct gate decisions per question (runs of one label) median {c.median():g}, min {c.min()}, max {c.max()}", flush=True)
    print(f"SCORED: {len(out)} rows of {out.sequence.nunique()} questions -> {s18.preds_csv()}", flush=True)


def stats_cmd(a):
    from eyehead.infer import load_ckpt
    plan = read_plan()
    span = {}
    for q in plan["questions"]:
        bounds = [0] + list(q["starts"]) + [q["K"]]
        span[q["pseudo"]] = sum(q["mp4_s"][s1 - 1] - q["mp4_s"][s0] for s0, s1 in zip(bounds, bounds[1:]))
    group = {q["pseudo"]: q["group"] for q in plan["questions"]}
    ck = load_ckpt(os.path.join(paths.ckpt_dir(), "fold1.pt"), "cpu")
    cols = ["sequence", "feat_frame_2", "n_gaze_in_gap", "n_oof_in_gap", "n_imu_in_gap", check_gate.EYE_COL, check_gate.IMU_COL]
    df = pd.read_csv(os.path.join(s18.folds_dir(), "test.csv"), usecols=cols).merge(
        pd.read_csv(s18.preds_csv(), usecols=["feat_frame_2", "pred"]), on="feat_frame_2", how="left", validate="one_to_one")
    if df["pred"].isna().any():
        sys.exit(f"{int(df['pred'].isna().sum())} rows have no prediction. Rerun the score step.")
    df["frame"] = df["feat_frame_2"].map(check_gate.pol.frame_no)
    df = df.sort_values(["sequence", "frame"]).reset_index(drop=True)
    table = {}
    for name in ["hdepic18", *sorted(set(group.values()))]:
        part = df if name == "hdepic18" else df[df.sequence.map(group) == name]
        per = pd.DataFrame([check_gate.video_summary(v, x["pred"].to_numpy().astype(int), x["n_gaze_in_gap"].to_numpy(),
                                                     x["n_oof_in_gap"].to_numpy(), x["n_imu_in_gap"].to_numpy())
                            for v, x in part.groupby("sequence", sort=False)])
        s = dict(check_gate.group_summary(per, part["pred"].to_numpy().astype(int)), **check_gate.input_stats(part, ck))
        seconds = float(sum(span[v] for v in per.video))
        s["marks_per_min"] = 60 * per["marks"].sum() / seconds if seconds > 0 else float("nan")
        s["seconds_p50"] = float(np.median([span[v] for v in per.video]))
        table[name] = s
    for ref in ("aea12", "nymeria"):
        try:
            rdf = check_gate.load_group(ref)
        except Exception as ex:
            print(f"   {ref}: not compared ({type(ex).__name__}: {str(ex)[:120]})", flush=True)
            continue
        per = pd.DataFrame([check_gate.video_summary(v, x["pred"].to_numpy().astype(int), x["n_gaze_in_gap"].to_numpy(),
                                                     x["n_oof_in_gap"].to_numpy(), x["n_imu_in_gap"].to_numpy())
                            for v, x in rdf.groupby("sequence", sort=False)])
        table[ref] = dict(check_gate.group_summary(per, rdf["pred"].to_numpy().astype(int)), **check_gate.input_stats(rdf, ck))
    t = pd.DataFrame(table)
    pd.set_option("display.width", 240)
    print(t.round(3).to_string(), flush=True)
    t.to_csv(os.path.join(s18.root(), "gate_stats.csv"))
    vor = table["hdepic18"]["vor_median_cos"]
    if not np.isfinite(vor) or vor > gate.VOR_CLEARLY_NEGATIVE:
        sys.exit(f"VOR CHECK FAILED: vor_median_cos {vor:.3f}, not clearly negative (AEA and Nymeria about -0.88). Stop and tell the user.")
    print(f"VOR CHECK OK: vor_median_cos {vor:.3f}. STOP: show the user this table and wait for their OK.", flush=True)


def ours_frames(labels, starts, K, fractions=(0.05, 0.1, 0.2)):
    """Ours' frames on one question at each budget: the budget rule with section 14's numbers (17 tokens a frame, 64-frame cap)."""
    from vlmeval import budget_rule as br
    return [len(br.keep_frames(labels, br.frame_limit(f, min(K, s18.K) * 81, 17, K, s18.K), starts=starts)[0]) for f in fractions]


def labels_by_question(folds, preds_path):
    from vlmeval import egoev
    clips, _ = egoev.gate_clips(folds, test_only=False)
    return egoev.video_labels(clips, pd.read_csv(preds_path))


def compare_cmd(a):
    """D2's exception against the plain rule on the same questions: the plain rule's 64 frames over the video, scored the same way."""
    from accuracy_tables import common
    from accuracy_tables.hpc.hdepic import plan18
    plan = read_plan()
    touched = [q for q in plan["questions"] if q["exception"]]
    if not touched:
        print("COMPARED: no question takes D2's exception", flush=True)
        return
    streams = Streams()
    plain = []
    for q in touched:
        v = q["videos"][0]
        mp4_ns, vrs_ns = data.read_time_csv(os.path.join(data.video_dir(v), f"{v}_mp4_to_vrs_time_ns.csv"))
        allf = plan18.plan17.distinct_frames(mp4_ns, 0.0, float(mp4_ns[-1]) / 1e9)
        idx = allf[common.uniform_indices(len(allf), q["K"])]
        plain.append(dict(q, pseudo=q["pseudo"] + "_plain", starts=[], forced=[], video_of=[0] * len(idx),
                          device_us=[float(d) / 1e3 for d in vrs_ns[idx]], K=len(idx)))
    cmp = os.path.join(s18.root(), "compare")
    score_rows(plain, os.path.join(cmp, "folds"), os.path.join(cmp, "preds.csv"), lambda q: pd.DataFrame(question_rows(q, streams)))
    lab_plain = labels_by_question(os.path.join(cmp, "folds"), os.path.join(cmp, "preds.csv"))
    lab = labels_by_question(s18.folds_dir(), s18.preds_csv())
    all_frames = np.sum([ours_frames(lab[q["pseudo"]], q["starts"], q["K"]) for q in plan["questions"]], axis=0)
    forced = np.sum([ours_frames(lab[q["pseudo"]], q["starts"], q["K"]) for q in touched], axis=0)
    plainf = np.sum([ours_frames(lab_plain[p["pseudo"]], [], p["K"]) for p in plain], axis=0)
    for f, x, y, t in zip((5, 10, 20), forced, plainf, all_frames):
        print(f"budget {f}%: ours' frames on the {len(touched)} questions {int(x)} with the query frame, {int(y)} without; over all "
              f"{len(plan['questions'])} questions {int(t)}, so {100 * (x - y) / t:+.3f}% of ours' frames", flush=True)
    print("COMPARED: every row keeps its frame count (the query frame takes a frame's place), so Uniform's, the baselines' and the "
          "no-gate control's FLOPs don't move; ours' move only by the gate's labels on one candidate per question", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def imu(a):
        t, found = gate.imu_check(read_plan()["videos"], paths.read_list("aea12"))
        cols = ["source", "video", "hz", "accel_mag", "accel_dir_cos_to_aea", "gyro_mag_p50"]
        print(t[[c for c in cols if c in t]].round(3).to_string(index=False), flush=True)
        if found:
            sys.exit("IMU DIFFERS FROM AEA: " + "; ".join(found) + ". Stop and tell the user.")
        print("IMU MATCHES AEA: right IMU, m/s^2 and rad/s, the same device axes", flush=True)

    for name, fn in (("imu", imu), ("rows", rows_cmd), ("score", score_cmd), ("stats", stats_cmd), ("compare", compare_cmd)):
        sub.add_parser(name).set_defaults(run=fn)
    a = ap.parse_args(argv)
    a.run(a)


if __name__ == "__main__":
    main()
