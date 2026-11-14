"""Ours v2, design and Stage 1 only (2026-11-14): a 3-ring foveated frame on SmolVLM2-2.2B, each ring encoded as one contiguous image.
Diagnostic dev work: nothing here feeds a results table, v1 doesn't change, and every output goes to $GV_ROOT/data/hdepic/v2/.

    python -u -m accuracy_tables.hpc.hdepic.v2 layouts              (CPU) the layouts' sizes, tokens and encoder GFLOPs, drawn on 3 frames
    python -u -m accuracy_tables.hpc.hdepic.v2 dev                  (CPU) the 940 dev questions (seed 1), their 9 frames each, the plan
    python -u -m accuracy_tables.hpc.hdepic.v2 frames --index <i>   (CPU, an array task) one dev video's frames
    python -u -m accuracy_tables.hpc.hdepic.v2 frames-check         (CPU) every dev frame there; meta.json per question
    python -u -m accuracy_tables.hpc.hdepic.v2 stage1               (B200) the variants on the dev questions, then the table

The rings, from the 1344 px source frame (the stored JPEG; the MP4 is 1408 px):
  periphery  the whole frame, shrunk to P px
  mid        a crop around the gaze at stock scale (384 px for the whole frame): M px from 1344 x M / 384 source px
  fovea      a smaller crop around the gaze at 2x stock scale: F px from 1344 x F / 768 source px
Every side is a multiple of 42 px (SigLIP's 14 px patches x SmolVLM's 3 x 3 pixel shuffle) and at most 384 px, so each ring is a normal
SmolVLM image of (side / 42)^2 tokens through SmolVLM's own encoder and position table, never cell by cell. A crop is centred on the gaze
and slid inside the frame at an edge (never padded). No periphery token is dropped under a finer ring: the LLM reads each image's tokens in
raster order, and a hole would break that grid. In the prompt each frame keeps its one "Frame at <TIME ...>:" label, followed by its rings
as consecutive images (periphery, mid, fovea), each with SmolVLM's own single-image markers. The gaze, the 1 s fixation and the centred
fallback are v1's.

Stage 1 (dev, diagnostic): 940 fine_grained action_recognition questions outside section 18's 2,000-question sample, on videos with gaze,
spread over videos with seed 1; each question's 9 frames are Uniform K' = 9 of its section 18 candidates (D1 and D2), no gate, no budget.
Variants: stock 81, blind, avgpool 4x4 (16), L1, L2, L3 at the gaze, L2 at the centre, and the periphery alone at L2's token count.
"""
import argparse
import collections
import json
import os
import shutil
import sys

import numpy as np

from accuracy_tables.hpc.hdepic import data, groups, s18

SOURCE = 1344                          # the stored frames' side (section 17 and 18's JPEGs)
STOCK_SIDE = 384                       # SmolVLM's stock input for the whole frame
BASE = 42
DEV_N, DEV_SEED, DEV_FRAMES = 940, 1, 9
LAYOUTS = {                            # (ring, side px, scale against stock; None = the whole frame)
    "L1": (("periphery", 252, None), ("mid", 84, 1.0), ("fovea", 84, 2.0)),
    "L2": (("periphery", 252, None), ("mid", 126, 1.0), ("fovea", 168, 2.0)),
    "L3": (("periphery", 294, None), ("mid", 168, 1.0), ("fovea", 210, 2.0)),
    "P64": (("periphery", 336, None),),            # the periphery alone at about L2's tokens: is the fovea better than a sharper frame?
}
RING_CONDITIONS = {"L1_gaze": ("L1", "gaze"), "L2_gaze": ("L2", "gaze"), "L3_gaze": ("L3", "gaze"), "L2_centre": ("L2", "centre"),
                   "P64_periphery_only": ("P64", "gaze")}
# SmolVLM2-2.2B's SigLIP and connector (read from the model's config when it's there)
VISION = dict(layers=27, d=1152, ff=4304, patch=14, shuffle=3, llm=2048)


def root(*parts):
    d = os.path.join(os.environ["GV_ROOT"], "data", "hdepic", "v2", *parts)
    data.refuse_inside_raw(d)
    return d


# ---------------------------------------------------------------------------- the layouts
def tokens(side):
    return (side // BASE) ** 2


def layout_tokens(name):
    return sum(tokens(s) for _, s, _ in LAYOUTS[name])


def image_gflops(side, v=VISION):
    """Encoder + connector GFLOPs of one side x side image (2 FLOPs a multiply-add): SigLIP's layers, attention and patch embedding,
    then the connector's projection of each 3 x 3 patch group."""
    n = (side / v["patch"]) ** 2
    t = (side / (v["patch"] * v["shuffle"])) ** 2
    linear = 2 * v["layers"] * (4 * v["d"] ** 2 + 2 * v["d"] * v["ff"]) * n
    attention = 2 * v["layers"] * 2 * v["d"] * n * n
    embed = 2 * v["patch"] ** 2 * 3 * v["d"] * n
    connector = 2 * (v["shuffle"] ** 2 * v["d"]) * v["llm"] * t
    return (linear + attention + embed + connector) / 1e9


def layout_gflops(name, v=VISION):
    return sum(image_gflops(s, v) for _, s, _ in LAYOUTS[name])


def ring_boxes(gaze, name, W=SOURCE, H=SOURCE):
    """[(ring, side, (x0, y0, x1, y1) in source px)]: the periphery is the whole frame; a crop at scale k covers side x W / (384 k)
    source px, centred on the gaze and slid inside the frame."""
    gx, gy = gaze if gaze is not None else (0.5, 0.5)
    out = []
    for ring, side, k in LAYOUTS[name]:
        if k is None:
            out.append((ring, side, (0, 0, W, H)))
            continue
        src = int(round(side * W / (STOCK_SIDE * k)))
        x0 = int(round(min(max(gx * W - src / 2, 0), W - src)))
        y0 = int(round(min(max(gy * H - src / 2, 0), H - src)))
        out.append((ring, side, (x0, y0, x0 + src, y0 + src)))
    return out


def ring_images(x_src01, boxes):
    """Each ring as SmolVLM gets it: its box cut from the source (3, H, W) in [0, 1] and resized to its side (bicubic, antialiased), the
    source pixels themselves when the box is already that size."""
    import torch.nn.functional as F
    out = []
    for _, side, (x0, y0, x1, y1) in boxes:
        crop = x_src01[:, y0:y1, x0:x1]
        if crop.shape[-1] == side and crop.shape[-2] == side:
            out.append(crop.clone())
        else:
            out.append(F.interpolate(crop[None].float(), size=(side, side), mode="bicubic", antialias=True, align_corners=False)[0].clamp(0, 1))
    return out


def vision_config(model_id):
    """VISION from the model's own config, when it can be read offline."""
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(model_id)
        vc = c.vision_config
        return dict(layers=vc.num_hidden_layers, d=vc.hidden_size, ff=vc.intermediate_size, patch=vc.patch_size,
                    shuffle=getattr(c, "scale_factor", 3), llm=c.text_config.hidden_size)
    except Exception:
        return VISION


def layouts_table(v=VISION):
    stock = image_gflops(378, v)
    rows = []
    for name in LAYOUTS:
        rings = ", ".join(f"{r} {s} px ({tokens(s)} tok" + ("" if k is None else f", {100 * s / (STOCK_SIDE * k):.0f}% of the frame side")
                          + ")" for r, s, k in LAYOUTS[name])
        rows.append(f"{name}: {rings} = {layout_tokens(name)} tokens a frame, {layout_gflops(name, v):.1f} encoder GFLOPs "
                    f"({100 * layout_gflops(name, v) / stock:.0f}% of a stock frame's {stock:.1f}; tokens {100 * layout_tokens(name) / 81:.0f}%)")
    return rows, stock


def draw_layouts(frames, out):
    """Per frame: each layout's boxes on the frame, and the frame as the model gets it (each ring's image put back over its box, its
    42 px token cells drawn), with the gaze dot and the token count."""
    import cv2
    import torch
    T = 640
    made = []
    for i, (path, gaze, label) in enumerate(frames):
        bgr = cv2.imread(path)
        x_src = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255
        H, W = bgr.shape[:2]
        f = T / W
        rows = []
        for name in LAYOUTS:
            boxes = ring_boxes(gaze, name, W, H)
            imgs = ring_images(x_src, boxes)
            left = cv2.resize(bgr, (T, int(H * f)), interpolation=cv2.INTER_AREA)
            right = np.zeros_like(left)
            colours = {"periphery": (200, 200, 200), "mid": (0, 220, 255), "fovea": (0, 0, 255)}
            for (ring, side, (x0, y0, x1, y1)), img in zip(boxes, imgs):
                X0, Y0, X1, Y1 = (int(round(c * f)) for c in (x0, y0, x1, y1))
                rgb = (img.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)[:, :, ::-1]
                right[Y0:Y1, X0:X1] = cv2.resize(rgb, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_NEAREST)
                step = (X1 - X0) / (side // BASE)
                for k in range(side // BASE + 1):
                    cv2.line(right, (int(X0 + k * step), Y0), (int(X0 + k * step), Y1 - 1), colours[ring], 1)
                    cv2.line(right, (X0, int(Y0 + k * step)), (X1 - 1, int(Y0 + k * step)), colours[ring], 1)
                if ring != "periphery":
                    cv2.rectangle(left, (X0, Y0), (X1, Y1), colours[ring], 2)
            g = gaze if gaze is not None else (0.5, 0.5)
            for panel in (left, right):
                cv2.circle(panel, (int(g[0] * T), int(g[1] * H * f)), 6, (0, 255, 0), -1)
            text = (f"{name}: " + " + ".join(f"{r} {tokens(s)}" for r, s, _ in LAYOUTS[name]) + f" = {layout_tokens(name)} tokens, "
                    f"{layout_gflops(name):.0f} GFLOPs")
            pad = np.full((26, 2 * T, 3), 255, np.uint8)
            cv2.putText(pad, text + f"   ({label})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            rows.append(np.vstack([np.hstack([left, right]), pad]))
        path_out = os.path.join(out, f"frame_{i + 1}.jpg")
        cv2.imwrite(path_out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
        made.append(path_out)
    return made


def layouts_cmd(a):
    from accuracy_tables.hpc.hdepic.load import gaze_at, gaze_track
    from vlmeval import smolvlm_fovea as sm
    v = vision_config(sm.MODEL_CANDIDATES[0])
    rows, stock = layouts_table(v)
    print(f"--- v2 layouts (diagnostic design); a stock frame: 81 tokens, {stock:.1f} encoder GFLOPs; v1's 17-token frame about 130", flush=True)
    for line in rows:
        print("LAYOUT " + line, flush=True)
    with open(s18.plan_json()) as f:
        plan = json.load(f)
    picks, seen = [], set()
    for q in plan["questions"]:
        if q["type"] != "action_recognition" or q["videos"][0] in seen:
            continue
        j = q["K"] // 2
        video = q["videos"][q["video_of"][j]]
        g, _ = gaze_at(gaze_track(s18.gaze_npz(video)), q["device_us"][j] / 1e6)
        path = os.path.join(s18.frames_dir(q["pseudo"]), f"frame_{j:05d}.jpg")
        if g is not None and os.path.exists(path):
            picks.append((path, g, f"{q['pseudo']} frame {j}"))
            seen.add(q["videos"][0])
        if len(picks) == 3:
            break
    out = root("layouts")
    os.makedirs(out, exist_ok=True)
    made = draw_layouts(picks, out)
    print(f"LAYOUTS WRITTEN: {made} (HD-EPIC images: never commit them). STOP: the user approves the layouts.", flush=True)


# ---------------------------------------------------------------------------- the dev questions and their frames
def dev_pool(loaded, sample):
    """fine_grained action_recognition questions outside the sample, every segment on a video with gaze, no parse error."""
    t = "action_recognition"
    taken = set(sample.get(t, []))
    out = []
    for q in loaded[groups.SAMPLE_GROUP].get(t, []):
        if q["qid"] in taken or q["parse_error"] or not q["segments"]:
            continue
        if any(v in data.NO_GAZE_VIDEOS or not data.gaze_csv(v) for v in q["videos"]):
            continue
        out.append(q)
    return out


def choose_dev(pool, n=DEV_N, seed=DEV_SEED):
    """n question ids spread over videos (a question's first video), one generator seeded with 1."""
    byv = collections.defaultdict(list)
    for q in pool:
        byv[q["videos"][0]].append(q["qid"])
    return groups.spread_over_videos(byv, n, np.random.default_rng(seed))


def dev_entry(e, i):
    """A section 18 plan entry cut to its Uniform K' = 9 frames (common.uniform_indices over its candidates)."""
    from accuracy_tables import common
    sel = common.uniform_indices(e["K"], DEV_FRAMES)
    pick = lambda key: [e[key][j] for j in sel]
    return dict(qid=i, hdepic_qid=e["hdepic_qid"], group=e["group"], type=e["type"], pseudo=f"hd18dev_{i:05d}", videos=e["videos"],
                K_candidates=e["K"], n=len(sel), positions=sel, video_of=pick("video_of"), mp4_frames=pick("mp4_frames"),
                mp4_s=pick("mp4_s"), device_us=pick("device_us"), video_n=pick("video_n"), n_choices=e["n_choices"])


def dev_cmd(a):
    import pandas as pd
    from accuracy_tables.hpc.hdepic import plan as plan17
    from accuracy_tables.hpc.hdepic import plan18
    loaded, _ = groups.load_groups()
    fg = [q for qs in loaded[groups.SAMPLE_GROUP].values() for q in qs]
    sample, quota = groups.stratified_sample(fg)
    groups.write_or_check_sample(sample, quota, len(fg))
    pool = dev_pool(loaded, sample)
    cache, info = {}, {}
    for v in sorted({s["video"] for q in pool for s in q["segments"]}):
        mp4_ns, vrs_ns = data.read_time_csv(os.path.join(data.video_dir(v), f"{v}_mp4_to_vrs_time_ns.csv"))
        cache[v] = dict(mp4_ns=mp4_ns, vrs_ns=vrs_ns)
        info[v] = plan17.video_info(v)
        cache[v]["gaze_us"] = pd.read_csv(info[v]["gaze_csv"], usecols=["tracking_timestamp_us"])["tracking_timestamp_us"].to_numpy(np.float64)
        info[v]["gaze_hz"] = data.rate_hz(cache[v]["gaze_us"])
    entries = {}
    for q in pool:
        e, _ = plan18.plan_question(q, 0, cache)
        if e is not None:
            entries[q["qid"]] = e
    ids = choose_dev([q for q in pool if q["qid"] in entries])
    path = root("dev_ids.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = dict(type="action_recognition", n=len(ids), seed=DEV_SEED, outside=groups.sample_json(), qids=ids)
    if os.path.exists(path):
        with open(path) as f:
            if json.load(f).get("qids") != ids:
                sys.exit(f"DEV IDS DIFFER from {path}: they never change. Stop and tell the user.")
        state = "unchanged"
    else:
        with open(path, "w") as f:
            json.dump(body, f)
        state = "written"
    dev = [dev_entry(entries[q], i) for i, q in enumerate(ids)]
    videos = sorted({v for e in dev for v in e["videos"]})
    plan = dict(rule="v2 Stage 1 dev: Uniform K' = 9 of section 18's candidates", videos=videos, video_info={v: info[v] for v in videos},
                questions=dev)
    with open(root("dev_plan.json"), "w") as f:
        json.dump(plan, f)
    missing = [v for v in videos if not os.path.exists(s18.gaze_npz(v))]
    print(f"dev pool: {len(pool)} action_recognition questions outside the sample on videos with gaze; {len(entries)} with candidates", flush=True)
    print(f"DEV QUESTIONS: {len(ids)} (seed {DEV_SEED}) on {len(videos)} videos, {state}: {path}; {len(missing)} videos still need their gaze "
          "projected (the aria step that follows does it)", flush=True)


def dev_frames_dir(pseudo):
    return root("frames", pseudo)


def frame_index():
    """{(video, MP4 frame): (path, decoder s)} of every frame sections 17 and 18 already extracted."""
    from accuracy_tables.hpc.hdepic import frames18
    out = frames18.section17_frames()
    try:
        with open(s18.plan_json()) as f:
            plan = json.load(f)
    except FileNotFoundError:
        return out
    for q in plan["questions"]:
        meta = os.path.join(s18.frames_dir(q["pseudo"]), "meta.json")
        if os.path.exists(meta):
            with open(meta) as f:
                m = json.load(f)
            for j, (vi, fr, dec) in enumerate(zip(m["video_of"], m["mp4_frames"], m["decoder_s"])):
                out.setdefault((m["videos"][vi], int(fr)), (os.path.join(m["dir"], f"frame_{j:05d}.jpg"), float(dec)))
    return out


def extract_video(plan, video, old, folder=None):
    """One video's dev frames: hard-linked from sections 17 and 18 where they exist, the rest decoded in order (exact frame numbers).
    folder(clip) -> its frames folder (default: v2's dev folders; the sweep passes its own)."""
    import cv2
    folder = folder or dev_frames_dir
    from accuracy_tables.hpc.hdepic.frames import QUALITY, SIZE
    wanted = collections.defaultdict(list)
    for q in plan["questions"]:
        part = os.path.join(folder(q["pseudo"]), f"part_{video}.json")
        if video not in q["videos"] or os.path.exists(part):
            continue
        k = q["videos"].index(video)
        for j, (vi, f) in enumerate(zip(q["video_of"], q["mp4_frames"])):
            if vi == k:
                wanted[int(f)].append((q, j))
    decoder, decoded, linked = {}, 0, 0
    for qs in wanted.values():
        for q, _ in qs:
            os.makedirs(folder(q["pseudo"]), exist_ok=True)
    file_of = lambda q, j: os.path.join(folder(q["pseudo"]), f"frame_{j:05d}.jpg")

    def link(src, dst):
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    todo = sorted(f for f in wanted if (video, f) not in old)
    for f in wanted:
        if (video, f) in old:
            src, decoder[f] = old[video, f]
            for q, j in wanted[f]:
                link(src, file_of(q, j))
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
                cv2.imwrite(file_of(q0, j0), small, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                decoded += 1
                for q, j in rest:
                    link(file_of(q0, j0), file_of(q, j))
            elif not cap.grab():
                break
            i += 1
        cap.release()
        if i <= last:
            raise SystemExit(f"{video}: the MP4 ended at frame {i}, before planned frame {last}. Stop and tell the user.")
    parts = collections.defaultdict(dict)
    for f, qs in wanted.items():
        for q, j in qs:
            parts[q["pseudo"]][str(j)] = decoder[f]
    for pseudo, d in parts.items():
        with open(os.path.join(folder(pseudo), f"part_{video}.json"), "w") as fh:
            json.dump(d, fh)
    print(f"   {video}: {decoded} frames decoded, {linked} hard-linked from sections 17 and 18, for {len(parts)} questions", flush=True)


def frames_check(plan, tolerance=0.05):
    """-> (problems, frames): every dev question's frames and decoder times; writes its meta.json."""
    problems, n = [], 0
    for q in plan["questions"]:
        dec = {}
        for v in q["videos"]:
            p = os.path.join(dev_frames_dir(q["pseudo"]), f"part_{v}.json")
            if os.path.exists(p):
                with open(p) as f:
                    dec.update({int(k): float(x) for k, x in json.load(f).items()})
        files = [os.path.join(dev_frames_dir(q["pseudo"]), f"frame_{j:05d}.jpg") for j in range(q["n"])]
        if sorted(dec) != list(range(q["n"])) or not all(os.path.exists(p) for p in files):
            problems.append(f"{q['pseudo']}: {sum(os.path.exists(p) for p in files)} of {q['n']} frames, decoder times for {len(dec)}")
            continue
        off = max(abs(dec[j] - q["mp4_s"][j]) for j in range(q["n"]))
        if off > tolerance:
            problems.append(f"{q['pseudo']}: the decoder's frame times are up to {off:.3f} s from the time CSV's")
            continue
        meta = dict(n=q["n"], dir=dev_frames_dir(q["pseudo"]), ts=q["mp4_s"], device_us=q["device_us"], mp4_frames=q["mp4_frames"],
                    video_of=q["video_of"], videos=q["videos"], video_n=q["video_n"], starts=[], forced=[],
                    decoder_s=[dec[j] for j in range(q["n"])])
        with open(os.path.join(dev_frames_dir(q["pseudo"]), "meta.json"), "w") as f:
            json.dump(meta, f)
        n += q["n"]
    return problems, n


# ---------------------------------------------------------------------------- Stage 1
def dev_items():
    """The dev questions as scoring items: qid, video, question, options, answer, frames (v1's frame dicts: stamp, gaze, trace)."""
    from accuracy_tables.hpc.hdepic.load import gaze_at, gaze_track
    from vlmeval.qwen_mcq import Stamp
    with open(root("dev_plan.json")) as f:
        plan = json.load(f)
    folder = data.questions_dir()
    files, _ = groups.question_files(folder)
    raw, tracks, items = {}, {}, []
    for q in plan["questions"]:
        key = (q["group"], q["type"])
        if key not in raw:
            with open(os.path.join(folder, files[q["group"]][q["type"]])) as f:
                raw[key] = json.load(f)
        r = raw[key][q["hdepic_qid"]] if isinstance(raw[key], dict) else raw[key][int(q["hdepic_qid"])]
        with open(os.path.join(dev_frames_dir(q["pseudo"]), "meta.json")) as f:
            m = json.load(f)
        frames = []
        for j in range(m["n"]):
            v = m["videos"][m["video_of"][j]]
            if v not in tracks:
                tracks[v] = gaze_track(s18.gaze_npz(v))
            g, trace = gaze_at(tracks[v], m["device_us"][j] / 1e6)
            frames.append(dict(key=(q["pseudo"], j), path=os.path.join(m["dir"], f"frame_{j:05d}.jpg"),
                               t=Stamp(m["ts"][j], s18.stamp(m["ts"][j], m["video_n"][j])), gaze=g, gaze_trace=trace))
        items.append(dict(qid=int(q["qid"]), video=q["pseudo"], question=r["question"], options=list(r["choices"]),
                          answer="ABCDE"[int(r["correct_idx"])], frames=frames))
    return sorted(items, key=lambda it: (it["video"], it["qid"]))


def ring_messages(build_messages, it, n_rings, size):
    """v1's prompt for the frames, with each frame's image followed by its other rings as images (one label a frame)."""
    msgs = build_messages(it["question"], it["options"], [(fr["path"], fr["t"]) for fr in it["frames"]], size)
    content = []
    for c in msgs[0]["content"]:
        content.append(c)
        if c["type"] == "image":
            content += [dict(c) for _ in range(n_rings - 1)]
    return [dict(msgs[0], content=content)]


def run_rings(ad, items, conditions, out_path, workers=16, log_every=50):
    """Scores items under ring conditions {name: (layout, 'gaze' | 'centre')}; resumable, one record a (condition, question)."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    import torch
    from PIL import Image
    from accuracy_tables.models import build_messages
    from vlmeval import smolvlm_fovea as sm
    ip = ad.proc.image_processor
    mean, std = ip.image_mean, ip.image_std
    done = set()
    if os.path.exists(out_path):
        done = {(r["condition"], r["qid"]) for r in (json.loads(x) for x in open(out_path) if x.strip())}
    todo = [(it, c) for it in items for c in conditions if (c, it["qid"]) not in done]
    print(f"{len(done)} already scored, {len(todo)} to go -> {out_path}", flush=True)
    load = lambda fr: torch.from_numpy(np.asarray(Image.open(fr["path"]).convert("RGB"))).permute(2, 0, 1).float() / 255
    t_start, n_new = time.time(), 0
    with open(out_path, "a") as fh, ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        src, src_q = {}, None
        for it, c in todo:
            if it["qid"] != src_q:
                src, src_q = dict(zip((fr["key"] for fr in it["frames"]), pool.map(load, it["frames"]))), it["qid"]
            layout, mode = conditions[c]
            t0 = time.time()
            feats, counts, missing = [], [], 0
            for fr in it["frames"]:
                g = (0.5, 0.5) if mode == "centre" else fr["gaze"]
                missing += mode != "centre" and g is None
                x = src[fr["key"]]
                boxes = ring_boxes(g, layout, x.shape[-1], x.shape[-2])
                for img in ring_images(x, boxes):
                    f = sm.encode_pixels(ad.model, sm.normalize(img, mean, std))
                    feats.append(f)
                    counts.append(len(f))
            msgs = ring_messages(build_messages, it, len(LAYOUTS[layout]), sm.IMG * sm.IMG)
            ids = sm.prompt_ids(ad.proc, msgs, counts, sm.TOKENIZE["add_special_tokens"])
            p = sm.letter_probs(ad.model, sm.inputs_from_ids(ad.model, ids, feats), ad.letter_ids)
            err = None if np.isfinite(p).all() else "nan"
            pred = "ABCDE"[int(np.nanargmax(p))] if np.isfinite(p).any() else "?"
            fh.write(json.dumps(dict(condition=c, qid=it["qid"], video=it["video"], answer=it["answer"], pred=pred,
                                     correct=int(pred == it["answer"]), p=[round(float(v), 4) for v in p], n_frames=len(it["frames"]),
                                     n_image_tokens=int(sum(counts)), n_tokens=int(ids.shape[1]), n_gaze_missing=int(missing),
                                     layout=layout, gaze=mode, sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left  ({c})", flush=True)


def wilson_line(name, path, gflops):
    from accuracy_tables.hpc.hdepic.diag18 import wilson
    recs = [json.loads(x) for x in open(path) if x.strip()]
    recs = [r for r in recs if r.get("error") is None]
    acc, lo, hi = wilson(sum(r["correct"] for r in recs), len(recs))
    tpf = np.mean([r.get("n_image_tokens", 0) / max(1, r.get("n_frames", 0)) for r in recs]) if recs and "n_frames" in recs[0] else 0.0
    g = "-" if gflops is None else f"{gflops:.1f}"
    return f"STAGE 1 {name}: n {len(recs)}, accuracy {acc:.2f}% [{lo:.2f}, {hi:.2f}], {tpf:.1f} tokens a frame, {g} encoder GFLOPs a frame"


def stage1_cmd(a):
    from accuracy_tables import run
    from accuracy_tables.models import ADAPTERS, build_messages
    from vlmeval import avgpool as ap
    from vlmeval import smolvlm_fovea as sm
    items = dev_items()
    ad = ADAPTERS["smolvlm"](None, dict(run=dict(prefetch_workers=16))).load()
    fr0 = items[0]["frames"][len(items[0]["frames"]) // 2]
    ok, rep = ad.parity(fr0, items[0]["question"], items[0]["options"])
    for k, v in rep.items():
        print(f"   {k:<44} {v}", flush=True)
    if not ok:
        sys.exit("parity failed: nothing scored. Stop and tell the user.")
    print("parity OK", flush=True)
    out = root("stage1")
    os.makedirs(out, exist_ok=True)
    ad.score(items, {"stock_81": ("uniform", sm.IMG)}, os.path.join(out, "stock_81.jsonl"))
    run.run_blind(ad, items, os.path.join(out, "blind.jsonl"))
    ap.run_questions(ad.avgpool_backend(), items, {"avgpool_4x4_16": dict(side=4)}, os.path.join(out, "avgpool_4x4_16.jsonl"), build_messages,
                     prefetch_workers=16)
    for name, cond in RING_CONDITIONS.items():
        run_rings(ad, items, {name: cond}, os.path.join(out, f"{name}.jsonl"))
    v = vision_config(sm.MODEL_CANDIDATES[0])
    gf = dict(stock_81=image_gflops(378, v), blind=None, avgpool_4x4_16=image_gflops(378, v),
              **{name: layout_gflops(cond[0], v) for name, cond in RING_CONDITIONS.items()})
    print(f"\n--- v2 Stage 1 (diagnostic): {len(items)} dev questions, 9 Uniform frames each, no gate, no budget", flush=True)
    table = {}
    for name in ("stock_81", "blind", "avgpool_4x4_16", *RING_CONDITIONS):
        line = wilson_line(name, os.path.join(out, f"{name}.jsonl"), gf[name])
        print(line, flush=True)
        table[name] = line
    with open(os.path.join(out, "stage1.json"), "w") as f:
        json.dump(table, f)
    print("STAGE 1 DONE (diagnostic). STOP: the user picks one layout.", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("layouts")
    sub.add_parser("dev")
    s = sub.add_parser("frames")
    s.add_argument("--index", type=int, help="the dev video's position (default: $SLURM_ARRAY_TASK_ID)")
    sub.add_parser("frames-check")
    sub.add_parser("stage1")
    a = ap.parse_args(argv)
    if a.command == "layouts":
        layouts_cmd(a)
    elif a.command == "dev":
        dev_cmd(a)
    elif a.command == "stage1":
        stage1_cmd(a)
    else:
        with open(root("dev_plan.json")) as f:
            plan = json.load(f)
        if a.command == "frames":
            i = a.index if a.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
            if not 0 <= i < len(plan["videos"]):
                sys.exit(f"index {i}: the dev plan has {len(plan['videos'])} videos (0 to {len(plan['videos']) - 1})")
            extract_video(plan, plan["videos"][i], frame_index())
        else:
            problems, n = frames_check(plan)
            for p in problems[:40]:
                print("  ", p, flush=True)
            if problems:
                sys.exit(f"DEV FRAMES DIFFER: {len(problems)} problems. Rerun the frames array for those videos, or tell the user.")
            print(f"DEV FRAMES MATCH: {n} frames for {len(plan['questions'])} questions", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
