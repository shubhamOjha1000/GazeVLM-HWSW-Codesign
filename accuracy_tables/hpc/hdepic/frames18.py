"""Runbook section 18, step 3: exactly the planned candidate frames, read by MP4 frame number. CPU only, base environment.

    python -u -m accuracy_tables.hpc.hdepic.frames18 extract --index <i>     one planned video (a Slurm array task: plan order)
    python -u -m accuracy_tables.hpc.hdepic.frames18 check                   every question: its frames, by count and by time; writes meta.json

Frames are written as sections 9 and 17 write them (long side 1344 px, JPEG quality 92), to $GV_ROOT/data/hdepic/s18/frames/<clip>/
frame_00000.jpg ..., one folder per question in its candidates' order (a question can hold frames of several videos). A video frame is
decoded once: a frame section 17 already extracted is hard-linked from there (the same decoder and encoding give the same file), and a
frame two questions share is written once and hard-linked. Each video's task writes part_<video>.json per question (the decoder's time of
each of its frames), so a rerun skips what's done. check then writes each question's meta.json from its parts and the plan, compares every
frame's decoder time with the time CSV's, and ends with FRAMES MATCH PLAN.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

from accuracy_tables.hpc.hdepic import data, s18
from accuracy_tables.hpc.hdepic.frames import DECODER_TOLERANCE_S, QUALITY, SIZE


def read_plan():
    with open(s18.plan_json()) as f:
        return json.load(f)


def frame_file(pseudo, j):
    return os.path.join(s18.frames_dir(pseudo), f"frame_{j:05d}.jpg")


def part_file(pseudo, video):
    return os.path.join(s18.frames_dir(pseudo), f"part_{video}.json")


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def section17_frames():
    """{(video, MP4 frame): (path, decoder s)} of every frame section 17 extracted, from its plan and meta.json files."""
    out = {}
    try:
        plan = data.read_plan()
    except FileNotFoundError:
        return out
    for q in plan["questions"]:
        meta = os.path.join(data.frames_dir(q["pseudo"]), "meta.json")
        if not os.path.exists(meta):
            continue
        with open(meta) as f:
            m = json.load(f)
        for j, (f_, dec) in enumerate(zip(m["mp4_frames"], m.get("decoder_s", m["ts"]))):
            p = os.path.join(m["dir"], f"frame_{j:05d}.jpg")
            if os.path.exists(p):
                out.setdefault((q["video"], int(f_)), (p, float(dec)))
    return out


def wanted_on(plan, video):
    """{MP4 frame: [(question, position)]} of one video's planned frames, for the questions whose part isn't written yet."""
    out = {}
    for q in plan["questions"]:
        if video not in q["videos"] or os.path.exists(part_file(q["pseudo"], video)):
            continue
        k = q["videos"].index(video)
        for j, (vi, f) in enumerate(zip(q["video_of"], q["mp4_frames"])):
            if vi == k:
                out.setdefault(int(f), []).append((q, j))
    return out


def extract_video(plan, video, old=None):
    import cv2
    wanted = wanted_on(plan, video)
    if not wanted:
        print(f"   {video}: every question's frames are there", flush=True)
        return 0, 0
    old = section17_frames() if old is None else old
    decoder, decoded, linked = {}, 0, 0
    for qs in wanted.values():
        for q, _ in qs:
            os.makedirs(s18.frames_dir(q["pseudo"]), exist_ok=True)
    todo = sorted(f for f in wanted if (video, f) not in old)
    for f in sorted(wanted):
        if (video, f) in old:
            src, decoder[f] = old[video, f]
            for q, j in wanted[f]:
                link(src, frame_file(q["pseudo"], j))
                linked += 1
    if todo:
        cap = cv2.VideoCapture(os.path.join(data.video_dir(video), f"{video}.mp4"))
        want, last, i = set(todo), todo[-1], 0
        while i <= last:
            if i in want:
                ok, fr = cap.read()
                if not ok:
                    break
                decoder[i] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3
                h, w = fr.shape[:2]
                s = SIZE / max(h, w)
                small = cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                (q0, j0), *rest = wanted[i]
                cv2.imwrite(frame_file(q0["pseudo"], j0), small, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                decoded += 1
                for q, j in rest:
                    link(frame_file(q0["pseudo"], j0), frame_file(q["pseudo"], j))
            elif not cap.grab():
                break
            i += 1
        cap.release()
        if i <= last:
            raise SystemExit(f"{video}: the MP4 ended at frame {i}, before planned frame {last}. Stop and tell the user.")
    parts = {}
    for f, qs in wanted.items():
        for q, j in qs:
            parts.setdefault(q["pseudo"], {})[str(j)] = decoder[f]
    for pseudo, d in parts.items():
        path = part_file(pseudo, video)
        with open(path + ".tmp", "w") as fh:
            json.dump(d, fh)
        os.replace(path + ".tmp", path)
    print(f"   {video}: {decoded} frames decoded, {linked} hard-linked from section 17, for {len(parts)} questions", flush=True)
    return decoded, linked


def question_meta(q, decoder):
    return dict(n=q["K"], dir=s18.frames_dir(q["pseudo"]), ts=q["mp4_s"], device_us=q["device_us"], mp4_frames=q["mp4_frames"],
                video_of=q["video_of"], videos=q["videos"], video_n=q["video_n"], starts=q["starts"], forced=q["forced"],
                decoder_s=decoder, size=SIZE)


def check(plan):
    """-> (questions, files, bytes on disk, problems, worst decoder offset); writes meta.json for every complete question."""
    problems, files, inodes, worst = [], 0, {}, 0.0
    for q in plan["questions"]:
        dec = {}
        for v in q["videos"]:
            p = part_file(q["pseudo"], v)
            if not os.path.exists(p):
                problems.append(f"{q['pseudo']}: no part for {v}")
                continue
            with open(p) as f:
                dec.update({int(k): float(x) for k, x in json.load(f).items()})
        paths_ = [frame_file(q["pseudo"], j) for j in range(q["K"])]
        have = [p for p in paths_ if os.path.exists(p)]
        if len(have) != q["K"] or sorted(dec) != list(range(q["K"])):
            problems.append(f"{q['pseudo']}: {len(have)} of {q['K']} frames, decoder times for {len(dec)}")
            continue
        decoder = [dec[j] for j in range(q["K"])]
        off = float(np.abs(np.asarray(decoder) - np.asarray(q["mp4_s"])).max())
        worst = max(worst, off)
        if off > DECODER_TOLERANCE_S:
            problems.append(f"{q['pseudo']}: the decoder's frame times are up to {off:.3f} s from the time CSV's")
            continue
        meta = os.path.join(s18.frames_dir(q["pseudo"]), "meta.json")
        with open(meta + ".tmp", "w") as f:
            json.dump(question_meta(q, decoder), f)
        os.replace(meta + ".tmp", meta)
        files += len(have)
        for p in have:
            st = os.stat(p)
            inodes[st.st_dev, st.st_ino] = st.st_size
    return len(plan["questions"]), files, sum(inodes.values()), len(inodes), problems, worst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("extract")
    s.add_argument("--index", type=int, help="the planned video's position (default: $SLURM_ARRAY_TASK_ID)")
    sub.add_parser("check")
    a = ap.parse_args(argv)
    data.refuse_inside_raw(os.path.join(s18.root(), "frames"))
    plan = read_plan()
    if a.command == "extract":
        i = a.index if a.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        if not 0 <= i < len(plan["videos"]):
            sys.exit(f"index {i}: the plan has {len(plan['videos'])} videos (0 to {len(plan['videos']) - 1})")
        extract_video(plan, plan["videos"][i])
        return
    n, files, nbytes, unique, problems, worst = check(plan)
    print(f"the decoder's frame times are at most {worst:.4f} s from the time CSV's", flush=True)
    for p in problems[:40]:
        print("  ", p, flush=True)
    if problems:
        sys.exit(f"FRAMES DIFFER FROM PLAN: {len(problems)} problems. Rerun the frames array for those videos, or tell the user.")
    print(f"FRAMES MATCH PLAN: {files} frame files for {n} questions, {unique} distinct on disk, {nbytes / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
