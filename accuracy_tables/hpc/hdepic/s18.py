"""Runbook section 18: where its derived data live, the user's decisions D1 to D5 as constants, and the frame stamps. Standard library only.

Everything derived for section 18 is under $GV_ROOT/data/hdepic/s18/, apart from the projected gaze, which is per real video and shared with
section 17 ($GV_ROOT/data/hdepic/gaze/). Results go to results/hdepic_all_smolvlm_flops/.

The user's decisions (2026-11-14, before any accuracy; every row the same):
  D1  K = min(64, distinct MP4 frames in the span), section 17's rule; a single instant is widened by 1 s either side.
  D2  segments in time order, K split over them by duration (1 at least each), the gate run per segment and never across a gap. Except
      object_motion's object_movement_counting and stationary_object_localization (a whole video plus one query instant inside it): the
      frame at the query instant is always included, in every row, and the rest of K is spread over the video, no frame twice.
  D3  before every frame, in every row, its time in the tags' own format, <TIME hh:mm:ss.sss video n>, on that video's MP4 clock (through
      the time CSV), as "Frame at <TIME ...>:". Every <TIME> and <BBOX> tag in the question and choices stays verbatim; no box is drawn.
      A new prompt setting relative to section 14: a deviation, counted in the prompt fit and the FLOPs.
  D4  section 17's resampler.
  D5  the 67 questions on the 2 videos without gaze are excluded from every row (fine_grained 29, ingredient 20, recipe 18). Questions
      with a segment outside gaze coverage stay; a frame without gaze gets section 9's centred fovea.
"""
import os
import re

K = 64
AROUND_S = 1.0
WINDOW_US = 1_000_000
EXCEPTION_TYPES = (("object_motion", "object_movement_counting"), ("object_motion", "stationary_object_localization"))
D5_EXPECTED = {"fine_grained": 29, "ingredient": 20, "recipe": 18}
NOTES = {
    "object_motion": "people look at what they pick up and put down, so gaze-driven rows' fovea often holds the moving object: information "
                     "that follows from gaze",
    "ingredient": "a weight question's scale or label is often where the wearer looks, so gaze-driven rows' fovea can hold the number asked for",
}
SAMPLE_LABEL = "2,000-question stratified sample of 21,000"
STAMP = re.compile(r"^<TIME \d{2}:\d{2}:\d{2}\.\d{3} video \d+>$")


def root():
    return os.path.join(os.environ["GV_ROOT"], "data", "hdepic", "s18")


def plan_json():
    return os.path.join(root(), "plan.json")


def rows_csv(pseudo):
    return os.path.join(root(), "rows", f"{pseudo}.csv")


def folds_dir():
    return os.path.join(root(), "folds")


def preds_csv():
    return os.path.join(root(), "preds.csv")


def frames_dir(pseudo):
    return os.path.join(root(), "frames", pseudo)


def questions_json():
    return os.path.join(root(), "questions.json")


def sheets_dir():
    return os.path.join(root(), "sheets")


def gaze_npz(video):
    """Section 17's projected gaze for one real video, shared: the projection is per video, not per question."""
    return os.path.join(os.environ["GV_ROOT"], "data", "hdepic", "gaze", f"{video}.npz")


def results_dir():
    return os.path.join(os.environ["GV_ROOT"], "results", "hdepic_all_smolvlm_flops")


def pseudo_name(qid):
    return f"hd18_{int(qid):05d}"


def stamp(mp4_s, n):
    """<TIME hh:mm:ss.sss video n> for an MP4 time in seconds, in the tags' own format."""
    ms = int(round(float(mp4_s) * 1000))
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    return f"<TIME {h:02d}:{m:02d}:{rest // 1000:02d}.{rest % 1000:03d} video {int(n)}>"


def input_number(name, order):
    """A question input's video number: the number in its name ('video 2' -> 2), else its order among the question's videos (from 1)."""
    m = re.search(r"(\d+)\s*$", str(name))
    return int(m.group(1)) if m else int(order)


def video_clock(video):
    """A sortable recording start from an HD-EPIC video name (P01-20240202-110250 -> '20240202110250'), or None."""
    m = re.search(r"(\d{8})-(\d{6})$", str(video))
    return m.group(1) + m.group(2) if m else None
