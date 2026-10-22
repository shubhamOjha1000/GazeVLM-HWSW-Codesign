"""DyCoke TTM baseline (vlmeval/dycoke.py). CPU only, no models, no data.

  1. references in a window: odd offsets the frame before, even offsets the window's first
  2. a 4-frame window is DyCoke's TTM: f1 whole, f2 | f1, f3 | f1, f4 | f3 (least similar kept)
  3. longer windows and a shorter last window follow the same rules
  4. tokens kept per pruned frame (at least one) and the share a full window keeps
  5. run_questions: frames encoded once per video, one record per (condition, question), resumable
"""
import json

import numpy as np
import pytest
import torch

from vlmeval import dycoke as dy
from vlmeval import streamingtom as stm


def _frame(seed, n=8, d=4):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def _flip(f, idx):
    g = f.clone()
    g[idx] = -f[idx]
    return g


def test_references_in_a_window():
    assert [dy.reference(j) for j in range(8)] == [None, 0, 0, 2, 0, 4, 0, 6]


def test_window_of_four_is_dycokes_ttm():
    f1 = _frame(0)
    f2 = _flip(f1, [2, 5])                         # differs from f1 at 2, 5
    f3 = _flip(f1, [1, 3])                         # differs from f1 at 1, 3
    f4 = _flip(f3, [0, 4])                         # differs from f3 at 0, 4
    plans = dy.ttm_plans([f1, f2, f3, f4], window=4, prune=0.75)          # keeps int(0.25 * 8) = 2
    assert plans == [list(range(8)), [2, 5], [1, 3], [0, 4]]


def test_longer_window_and_short_last_window():
    frames = [_frame(s) for s in range(4)]
    f5 = _frame(10)
    frames += [f5, _flip(f5, [6, 7])]              # a last window of 2: f5 whole, f6 | f5
    plans = dy.ttm_plans(frames, window=4, prune=0.75)
    assert plans[4] == list(range(8)) and plans[5] == [6, 7]
    plans8 = dy.ttm_plans([_frame(s) for s in range(8)], window=8, prune=0.75)
    assert [len(p) for p in plans8] == [8] + [2] * 7 and all(p == sorted(p) for p in plans8)


def test_keep_count_and_retention():
    assert dy.keep_count(256, 0.98) == 5 and dy.keep_count(256, 0.93) == 17 and dy.keep_count(81, 0.99) == 1
    assert dy.retention(32, 0.98, 256) == pytest.approx(411 / 8192)          # ~5%
    assert dy.retention(32, 0.93, 256) == pytest.approx(783 / 8192)          # ~10%
    assert dy.retention(4, 0.7, 196) == pytest.approx(370 / 784)             # DyCoke's own setting on 196 tokens


class FakeBackend:
    def __init__(self):
        self.encoded = []

    def frame_state(self, fr):
        self.encoded.append(fr["key"])
        return dict(tokens=_frame(100 + fr["key"][1]))

    def pack(self, st, plan, beta):
        return (stm.apply_plan(st["tokens"], plan, beta),)

    def score(self, it, per, build_messages):
        p = np.full(5, 0.1)
        p["ABCDE".index(it["answer"])] = 0.6
        return p, 10 + sum(len(f[0]) for f in per)


def _items():
    fr = lambda v, k: dict(key=(v, k), path=f"{v}_{k}.jpg", t=float(k))
    return [dict(qid=q, video=v, question="q", options=list("abcde"), answer="D", frames=[fr(v, k) for k in range(6)])
            for q, v in ((1, "a"), (2, "a"), (3, "b"))]


def test_run_questions_records_caches_and_resumes(tmp_path):
    out = str(tmp_path / "dycoke.jsonl")
    conds = {"w4": dict(window=4, prune=0.75), "w2": dict(window=2, prune=0.75)}
    be = FakeBackend()
    dy.run_questions(be, _items(), conds, out, build_messages=None)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 6 and all(r["correct"] == 1 and r["error"] is None for r in rows)
    assert sorted(be.encoded) == sorted({(v, k) for v in "ab" for k in range(6)})   # once per frame and video
    w4 = next(r for r in rows if r["condition"] == "w4")
    w2 = next(r for r in rows if r["condition"] == "w2")
    assert w4["n_image_tokens"] == (8 + 3 * 2) + (8 + 2) and w4["kept_share"] == pytest.approx(24 / 48)
    assert w2["n_image_tokens"] == 3 * (8 + 2) and w4["n_tokens"] == 10 + 24
    assert w4["dycoke"] == dict(window=4, prune=0.75)
    dy.run_questions(FakeBackend(), _items(), conds, out, build_messages=None)
    assert len(open(out).read().splitlines()) == 6
