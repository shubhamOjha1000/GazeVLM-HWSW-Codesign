"""The two-view GazeVLM (vlmeval/gazevlm2.py). CPU only, no models, no data.

  1. the local view: a gaze-centred crop of alpha x the side, shifted inside the frame
  2. token selection as the authors': a uniform grid, then the tokens nearest the gaze until distinct
  3. budgets per view; Qwen positions (local-view tokens as finer cells at their place in the frame)
  4. run_questions: global view once per frame, local view once per (frame, alpha), records, resumable
"""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from vlmeval import gazevlm2 as gv


def test_crop_window_centre_edges_and_whole_frame():
    assert gv.crop_window((0.5, 0.5), 672) == ((168, 168, 336), (0.5, 0.5))
    (r0, c0, side), frac = gv.crop_window((0.05, 0.9), 672)                  # gaze near the bottom-left corner
    assert (r0, c0, side) == (336, 0, 336)                                    # shifted inside the frame
    assert frac == pytest.approx(((0.9 * 672 - 336) / 336, 0.05 * 672 / 336))
    assert gv.crop_window((0.3, 0.7), 672, alpha=1.0) == ((0, 0, 672), pytest.approx((0.7, 0.3)))


def test_gaze_token():
    assert gv.gaze_token((0.5, 0.5), 16) == (8, 8) and gv.gaze_token((1.0, 0.0), 16) == (15, 0)


def test_uniform_grid():
    assert gv.token_sampling_grid(16, 16, 16) == [(r, c) for r in (2, 6, 10, 14) for c in (2, 6, 10, 14)]
    assert gv.token_sampling_grid(16, 16, 4) == [(4, 4), (4, 12), (12, 4), (12, 12)]
    pts = gv.token_sampling_grid(24, 24, 100)
    assert len(pts) == 100 and len(set(pts)) == 100 and all(0 <= r < 24 and 0 <= c < 24 for r, c in pts)


def test_token_sampling_circle_follows_the_authors():
    sel = gv.token_sampling_circle(16, 16, 8, 8, 8, 0.5)                      # 4 on the grid, 4 nearest the gaze
    assert [tuple(r) for r in sel] == [(4, 4), (4, 12), (7, 8), (8, 7), (8, 8), (8, 9), (12, 4), (12, 12)]
    sel = gv.token_sampling_circle(16, 16, 4, 4, 8, 0.5)                      # the gaze on a grid point: one more
    assert {tuple(r) for r in sel} == {(4, 4), (4, 12), (12, 4), (12, 12), (3, 4), (4, 3), (4, 5), (5, 4)}
    assert len(gv.token_sampling_circle(16, 16, 8, 8, 4, 0.0)) == 4           # beta 0: the grid only
    assert {tuple(r) for r in gv.token_sampling_circle(16, 16, 8, 8, 5, 1.0)} == {(8, 8), (7, 8), (8, 7), (8, 9), (9, 8)}
    assert len(gv.token_sampling_circle(4, 4, 0, 0, 99, 0.5)) == 16 and len(gv.token_sampling_circle(4, 4, 0, 0, 0)) == 0


def test_select_and_view_budget():
    assert gv.select(16, (8, 8), 8, 0.5).tolist() == [4 * 16 + 4, 4 * 16 + 12, 7 * 16 + 8, 8 * 16 + 7, 8 * 16 + 8,
                                                     8 * 16 + 9, 12 * 16 + 4, 12 * 16 + 12]
    assert gv.view_budget(13, 256) == 6 and gv.view_budget(1000, 256) == 256 and gv.view_budget(1, 81) == 0


def test_qwen_cells_put_local_tokens_at_their_place():
    be = object.__new__(gv.Qwen25)
    be.side, be.px = 16, 448
    local = dict(tokens=torch.arange(256.0)[:, None] + 1000)
    glob = dict(tokens=torch.arange(256.0)[:, None])
    tok, cells, px = be.pack_views(local, glob, torch.tensor([0, 17]), torch.tensor([0]), (336, 0, 672), 1344)
    assert tok.flatten().tolist() == [1000.0, 1017.0, 0.0] and px == 448            # local view first
    assert cells.tolist() == [[112.0, 0.0, 14.0], [126.0, 14.0, 14.0], [0.0, 0.0, 28.0]]


def test_generic_pack_concatenates_local_then_global():
    be = object.__new__(gv.InternVL)
    tok, = be.pack_views(dict(tokens=torch.ones(4, 2)), dict(tokens=torch.zeros(4, 2)),
                         torch.tensor([1, 2]), torch.tensor([3]), (0, 0, 2), 4)
    assert tok.sum(1).tolist() == [2.0, 2.0, 0.0]


# ---------------------------------------------------------------------------- the run loop
class FakeBackend(gv._Views):
    side = 4

    def __init__(self):
        self.encoded = []

    def frame_state(self, fr):
        size = Image.open(fr["path"]).size
        self.encoded.append((fr["key"], size))
        return dict(tokens=torch.full((16, 1), float(size[0])))

    def score(self, it, per, build_messages):
        p = np.full(5, 0.1)
        p["ABCDE".index(it["answer"])] = 0.6
        return p, 10 + sum(len(f[0]) for f in per)


def test_run_questions_two_views_per_frame_and_resumes(tmp_path):
    paths = {}
    for k in range(2):
        paths[k] = str(tmp_path / f"f{k}.png")
        Image.new("RGB", (64, 64), (10 * k, 0, 0)).save(paths[k])
    fr = lambda k, g: dict(key=("a", k), path=paths[k], t=float(k), gaze=g)
    items = [dict(qid=q, video="a", question="q", options=list("abcde"), answer="E",
                  frames=[fr(0, (0.5, 0.5)), fr(1, None)]) for q in (1, 2)]
    out = str(tmp_path / "gazevlm2.jsonl")
    be = FakeBackend()
    gv.run_questions(be, items, {"B=4": dict(tokens=4)}, out, build_messages=None)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 2 and all(r["correct"] == 1 and r["error"] is None for r in rows)
    assert sorted(be.encoded) == sorted([(("a", 0), (64, 64)), (("a", 0), (32, 32)),
                                         (("a", 1), (64, 64)), (("a", 1), (32, 32))])   # each view once
    r = rows[0]
    assert r["n_frames"] == 2 and r["n_encoder_images"] == 4 and r["n_image_tokens"] == 2 * 4
    assert r["n_gaze_missing"] == 1 and r["gazevlm2"] == dict(tokens=4, alpha=0.5, beta=0.5)
    gv.run_questions(FakeBackend(), items, {"B=4": dict(tokens=4)}, out, build_messages=None)
    assert len(open(out).read().splitlines()) == 2
