"""Runbook section 17, step 1c: gaze overlay sheets for the user to look at. CPU only, base environment.

    python -u -m accuracy_tables.hpc.hdepic.sheets [--videos 5] [--answers 20]

For 5 planned videos (at least one at 60 Hz when there is one, then the ones with the most "where looking" questions), a sheet of up
to 12 tiles: the MP4 frame at the end of a "where looking" span, with the projected gaze at that moment as a dot. Then an answer sheet:
up to 20 of those questions, each tile showing the gaze and, under it, the correct choice and the others, so the user sees where the gaze
lands against the answer. Prints, per video, the share of its projected gaze inside the frame. Writes $GV_ROOT/data/hdepic/sheets/
(it holds HD-EPIC images and text: never commit it).
"""
import argparse
import json
import os

import numpy as np

from accuracy_tables.hpc.hdepic import data

TILE, COLS = 448, 4


def read_frame(video, index):
    import cv2
    cap = cv2.VideoCapture(os.path.join(data.video_dir(video), f"{video}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def gaze_near(g, t_us, max_ms=60):
    """The projected gaze sample nearest t_us within max_ms -> (x, y, out of view) or None."""
    k = int(np.clip(np.searchsorted(g["t_us"], t_us), 0, len(g["t_us"]) - 1))
    best = min((k, max(k - 1, 0)), key=lambda i: abs(g["t_us"][i] - t_us))
    if abs(g["t_us"][best] - t_us) > max_ms * 1e3 or not np.isfinite(g["x"][best]):
        return None
    return float(g["x"][best]), float(g["y"][best]), bool(g["oof"][best])


def tile(frame, gaze, lines):
    import cv2
    img = cv2.resize(frame, (TILE, TILE), interpolation=cv2.INTER_AREA) if frame is not None else np.zeros((TILE, TILE, 3), np.uint8)
    if gaze is not None:
        x, y, oof = gaze
        cv2.circle(img, (int(x * TILE), int(y * TILE)), 10, (0, 0, 255) if not oof else (0, 255, 255), 3)
    pad = np.full((18 * len(lines) + 6, TILE, 3), 255, np.uint8)
    for i, text in enumerate(lines):
        safe = "".join(c if ord(c) < 128 else "?" for c in text)[:70]
        cv2.putText(pad, safe, (4, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, pad])


def sheet(tiles, path):
    import cv2
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((h - t.shape[0], TILE, 3), 255, np.uint8)]) for t in tiles]
    while len(tiles) % COLS:
        tiles.append(np.full((h, TILE, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def choose_videos(plan, n):
    ge = [q for q in plan["questions"] if q["type"] == "gaze_estimation"]
    count = {v: sum(q["video"] == v for q in ge) for v in plan["videos"]}
    hz = {v: plan["video_info"][v].get("gaze_hz") or 0 for v in plan["videos"]}
    ranked = sorted(plan["videos"], key=lambda v: (-count[v], v))
    fast = [v for v in ranked if hz[v] > 45]
    out = fast[:1] + [v for v in ranked if v not in fast[:1]]
    return out[:n]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=int, default=5)
    ap.add_argument("--answers", type=int, default=20)
    a = ap.parse_args(argv)
    out = data.sheets_dir()
    data.refuse_inside_raw(out)
    os.makedirs(out, exist_ok=True)
    plan = data.read_plan()
    raw = {}
    with open(os.path.join(data.questions_dir(), data.GAZE_TYPES["gaze_estimation"])) as f:
        raw = json.load(f)
    chosen = choose_videos(plan, a.videos)
    answer_tiles = []
    for v in chosen:
        with np.load(data.gaze_npz(v)) as z:
            g = {k: z[k] for k in ("t_us", "x", "y", "oof")}
        inside = np.isfinite(g["x"]) & ~g["oof"].astype(bool) & (g["x"] >= 0) & (g["x"] <= 1) & (g["y"] >= 0) & (g["y"] <= 1)
        qs = [q for q in plan["questions"] if q["video"] == v and q["type"] == "gaze_estimation"][:12]
        tiles = []
        for q in qs:
            fr = read_frame(v, q["mp4_frames"][-1])
            gz = gaze_near(g, q["device_us"][-1])
            tiles.append(tile(fr, gz, [f"{q['hdepic_qid']}  t {q['end_s']:.2f} s", "no gaze within 60 ms" if gz is None else
                                       f"gaze ({gz[0]:.2f}, {gz[1]:.2f}){' out of view' if gz[2] else ''}"]))
            if len(answer_tiles) < a.answers:
                r = raw[q["hdepic_qid"]]
                lines = [f"{q['hdepic_qid']}", f"answer: {r['choices'][int(r['correct_idx'])]}"]
                lines += [f"  other: {c}" for i, c in enumerate(r["choices"]) if i != int(r["correct_idx"])]
                answer_tiles.append(tile(fr, gz, lines))
        path = os.path.join(out, f"gaze_{v}.jpg")
        if tiles:
            sheet(tiles, path)
        print(f"   {v}: gaze {plan['video_info'][v].get('gaze_hz', float('nan')):.2f} Hz, {100 * inside.mean():.1f}% of projected gaze "
              f"inside the frame; {len(tiles)} tiles -> {path}", flush=True)
    if answer_tiles:
        sheet(answer_tiles, os.path.join(out, "answers_gaze_estimation.jpg"))
    print(f"SHEETS WRITTEN: {len(chosen)} videos and {len(answer_tiles)} answer tiles in {out}. STOP: the user looks at them.", flush=True)


if __name__ == "__main__":
    main()
