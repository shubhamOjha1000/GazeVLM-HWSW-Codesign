"""Runbook section 17, step 3: exactly the planned candidate frames, read by MP4 frame number. CPU only, base environment.

    python -u -m accuracy_tables.hpc.hdepic.frames extract --index <i>     one planned video (a Slurm array task: plan order)
    python -u -m accuracy_tables.hpc.hdepic.frames check                   every planned frame, by count and by time

A video is decoded once, in order; every planned frame number is kept, resized so its long side is 1344 px and written as JPEG
quality 92, as section 9 writes its frames (vlmeval.frames.extract_1fps). Each question gets its own folder,
$GV_ROOT/data/hdepic/frames/<clip>/frame_00000.jpg ..., and a meta.json written last, so a folder with meta.json is complete and a rerun
skips it. meta.json: n, dir, ts (MP4 seconds of each frame, from the time CSV), device_us, mp4_frames, decoder_s (the decoder's own
time of each kept frame), native_fps, size, orig_w, orig_h. check compares every frame by count, and by time against the decoder,
and ends with FRAMES MATCH PLAN.
"""
import argparse
import json
import os
import sys

import numpy as np

from accuracy_tables.hpc.hdepic import data

SIZE, QUALITY = 1344, 92
DECODER_TOLERANCE_S = 0.05              # the decoder's own time of a kept frame against the time CSV's, before check stops


def video_questions(plan, video):
    return [q for q in plan["questions"] if q["video"] == video]


def done(q):
    return os.path.exists(os.path.join(data.frames_dir(q["pseudo"]), "meta.json"))


def extract_video(plan, video):
    import cv2
    qs = [q for q in video_questions(plan, video) if not done(q)]
    if not qs:
        print(f"   {video}: every question's frames are there", flush=True)
        return 0
    wanted = {}
    for q in qs:
        for j, f in enumerate(q["mp4_frames"]):
            wanted.setdefault(int(f), []).append((q, j))
        os.makedirs(data.frames_dir(q["pseudo"]), exist_ok=True)
    cap = cv2.VideoCapture(os.path.join(data.video_dir(video), f"{video}.mp4"))
    native = cap.get(cv2.CAP_PROP_FPS)
    last, i, written, orig, decoder = max(wanted), 0, 0, None, {}
    while i <= last:
        if i in wanted:
            ok, fr = cap.read()
            if not ok:
                break
            decoder[i] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3
            h, w = fr.shape[:2]
            orig = (w, h)
            s = SIZE / max(h, w)
            small = cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            for q, j in wanted[i]:
                cv2.imwrite(os.path.join(data.frames_dir(q["pseudo"]), f"frame_{j:05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                written += 1
        elif not cap.grab():
            break
        i += 1
    cap.release()
    if i <= last:
        raise SystemExit(f"{video}: the MP4 ended at frame {i}, before planned frame {last}. Stop and tell the user.")
    for q in qs:
        meta = dict(n=q["K"], dir=data.frames_dir(q["pseudo"]), ts=q["mp4_s"], device_us=q["device_us"], mp4_frames=q["mp4_frames"],
                    decoder_s=[decoder[int(f)] for f in q["mp4_frames"]], native_fps=native, size=SIZE, orig_w=orig[0], orig_h=orig[1], video=video)
        path = os.path.join(data.frames_dir(q["pseudo"]), "meta.json")
        with open(path + ".tmp", "w") as f:
            json.dump(meta, f)
        os.replace(path + ".tmp", path)
    print(f"   {video}: {written} frames written for {len(qs)} questions", flush=True)
    return written


def check(plan):
    """-> (questions, frames, bytes, problems): every planned question's folder against the plan."""
    problems, frames, nbytes, worst = [], 0, 0, 0.0
    for q in plan["questions"]:
        d = data.frames_dir(q["pseudo"])
        meta_path = os.path.join(d, "meta.json")
        if not os.path.exists(meta_path):
            problems.append(f"{q['pseudo']}: no meta.json")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        files = [os.path.join(d, f"frame_{j:05d}.jpg") for j in range(q["K"])]
        have = [p for p in files if os.path.exists(p)]
        extra = len([n for n in os.listdir(d) if n.endswith(".jpg")]) - len(have)
        if len(have) != q["K"] or extra:
            problems.append(f"{q['pseudo']}: {len(have)} of {q['K']} frames, {extra} extra")
        if meta["n"] != q["K"] or meta["mp4_frames"] != q["mp4_frames"] or not np.allclose(meta["ts"], q["mp4_s"]):
            problems.append(f"{q['pseudo']}: meta.json's frames or times aren't the plan's")
        off = float(np.abs(np.asarray(meta.get("decoder_s", meta["ts"])) - np.asarray(q["mp4_s"])).max())
        worst = max(worst, off)
        if off > DECODER_TOLERANCE_S:
            problems.append(f"{q['pseudo']}: the decoder's frame times are up to {off:.3f} s from the time CSV's")
        frames += len(have)
        nbytes += sum(os.path.getsize(p) for p in have)
    return len(plan["questions"]), frames, nbytes, problems, worst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("extract")
    s.add_argument("--index", type=int, help="the planned video's position (default: $SLURM_ARRAY_TASK_ID)")
    sub.add_parser("check")
    a = ap.parse_args(argv)
    data.refuse_inside_raw(os.path.join(data.derived_root(), "frames"))
    plan = data.read_plan()
    if a.command == "extract":
        i = a.index if a.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        if not 0 <= i < len(plan["videos"]):
            sys.exit(f"index {i}: the plan has {len(plan['videos'])} videos (0 to {len(plan['videos']) - 1})")
        extract_video(plan, plan["videos"][i])
        return
    n, frames, nbytes, problems, worst = check(plan)
    print(f"the decoder's frame times are at most {worst:.4f} s from the time CSV's", flush=True)
    for p in problems[:40]:
        print("  ", p, flush=True)
    if problems:
        sys.exit(f"FRAMES DIFFER FROM PLAN: {len(problems)} problems. Rerun the frames array for those videos, or tell the user.")
    print(f"FRAMES MATCH PLAN: {frames} frames for {n} questions, {nbytes / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
