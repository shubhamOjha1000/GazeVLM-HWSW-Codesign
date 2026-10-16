"""AEA links JSON handling, and 1 fps frames from the AEA preview MP4s.

The frames are chosen with exactly the rule of src.common.aria_io.subsample_frames
(every round(native_fps)-th source frame), so frame k here is frame k of the gate's labels.
Each video's frames are cached on Drive as one .tar, so later sessions skip the download.
"""

import glob
import json
import os
import shutil
import tarfile
import time
import urllib.request

import cv2


# ---------------------------------------------------------------------------- links JSON
def link_ok(url, timeout=20):
    """True if a signed AEA download URL still works (they expire after ~14 days)."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status in (200, 206)
    except Exception:
        return False


def links_fresh(path):
    """(ok, message) for a links JSON: parses, and its first video URL still answers."""
    try:
        meta = json.load(open(path))
        seq, e = next(iter(meta["sequences"].items()))
        url = e["video_main_rgb"]["download_url"]
    except Exception as ex:
        return False, f"unreadable ({ex})"
    ok = link_ok(url)
    return ok, f"{len(meta['sequences'])} sequences, links {'OK' if ok else 'EXPIRED (403)'}"


def install_links(new_path, targets):
    """Copy a freshly uploaded links JSON over every older copy in `targets`."""
    for t in targets:
        os.makedirs(os.path.dirname(t), exist_ok=True)
        if os.path.exists(t):
            os.remove(t)
        shutil.copy2(new_path, t)


# ---------------------------------------------------------------------------- frames
def _download(url, path):
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)


def extract_1fps(mp4, out_dir, size=448, target_fps=1.0):
    """Keep every round(native/target_fps)-th frame, resized so the long side is `size`."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(mp4)
    native = cap.get(cv2.CAP_PROP_FPS) or 20.0
    step = max(1, int(round(native / target_fps)))
    ts, kept, i, orig = [], 0, 0, None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = fr.shape[:2]
            orig = (w, h)
            s = size / max(h, w)
            small = cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, f"frame_{kept:05d}.jpg"), small,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            ts.append(i / native)
            kept += 1
        i += 1
    cap.release()
    meta = dict(native_fps=native, step=step, n=kept, ts=ts, size=size,
                orig_w=orig[0] if orig else None, orig_h=orig[1] if orig else None)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"))
    return meta


def ensure_frames(seq, links, local_root, drive_root, raw_root="/content/raw", size=448):
    """Frames for one video in local_root/<seq>/, from the Drive cache or a fresh download.
    Returns the meta dict (with 'dir' added)."""
    d = os.path.join(local_root, seq)
    tar = os.path.join(drive_root, f"{seq}_px{size}.tar")
    if not os.path.exists(os.path.join(d, "meta.json")) and os.path.exists(tar):
        with tarfile.open(tar) as t:
            t.extractall(local_root)
    if not os.path.exists(os.path.join(d, "meta.json")):
        e = links["sequences"][seq]["video_main_rgb"]
        mp4 = os.path.join(raw_root, seq, e["filename"])
        os.makedirs(os.path.dirname(mp4), exist_ok=True)
        if not os.path.exists(mp4):
            t0 = time.time()
            _download(e["download_url"], mp4)
            print(f"   {seq}: downloaded {os.path.getsize(mp4)/1e6:.0f} MB "
                  f"in {time.time()-t0:.0f}s")
        extract_1fps(mp4, d, size)
        os.makedirs(drive_root, exist_ok=True)
        with tarfile.open(tar + ".part", "w") as t:
            t.add(d, arcname=seq)
        os.replace(tar + ".part", tar)
        shutil.rmtree(os.path.dirname(mp4), ignore_errors=True)
    meta = json.load(open(os.path.join(d, "meta.json")))
    meta["dir"] = d
    return meta


def frame_path(meta, k):
    return os.path.join(meta["dir"], f"frame_{k:05d}.jpg")
