"""DTD baseline (vlmeval/dtd.py). CPU only, no models, no data.

  1. the drop rule: cosine with the same position of the previous frame, absolute threshold or a share
  2. plans: the first frame whole, each frame compared with the previous frame as encoded
  3. the relative mode's retention formula
  4. run_questions: frames left without tokens leave the prompt, records, resumable
"""
import json

import numpy as np
import torch

from vlmeval import dtd
from vlmeval import streamingtom as stm


def test_dropped_absolute_and_relative():
    prev = torch.eye(4)
    e = torch.eye(4)
    curr = torch.stack([e[0], e[0], e[2] + e[3], e[3]])                       # cosines 1, 0, 0.707, 1
    assert dtd.dropped(prev, curr, 0.5, True).tolist() == [True, False, True, True]
    assert dtd.dropped(prev, curr, 0.8, True).tolist() == [True, False, False, True]
    assert dtd.dropped(prev, curr, 0.5, False).tolist() == [True, False, False, True]   # the 2 most similar
    assert not dtd.dropped(prev, curr, 0.0, False).any()


def test_plans_compare_with_the_previous_frame_as_encoded():
    A = torch.eye(4)
    plans = dtd.dtd_plans([A, A, -A], 0.5, True)
    assert plans == [[0, 1, 2, 3], [], [0, 1, 2, 3]]                           # frame 3 vs frame 2's own tokens


def test_relative_retention():
    assert abs(dtd.retention(0.96, 96, 256) - (256 + 95 * 11) / (96 * 256)) < 1e-12   # ~5.3%
    frames = list(torch.randn(5, 20, 8, generator=torch.Generator().manual_seed(0)))
    plans = dtd.dtd_plans(frames, 0.75, False)
    assert [len(p) for p in plans] == [20, 5, 5, 5, 5]
    assert dtd.retention(0.75, 5, 20) == sum(len(p) for p in plans) / 100


# ---------------------------------------------------------------------------- the run loop
class FakeBackend:
    def __init__(self):
        self.encoded = []

    def frame_state(self, fr):
        self.encoded.append(fr["key"])
        return dict(tokens=torch.eye(8))                                       # every frame the same 8 tokens

    def pack(self, st, plan, beta):
        return (stm.apply_plan(st["tokens"], plan, beta),)

    def score(self, it, per, build_messages):
        assert len(it["frames"]) == len(per)
        p = np.full(5, 0.1)
        p["ABCDE".index(it["answer"])] = 0.6
        return p, 10 + sum(len(f[0]) for f in per)


def _items():
    fr = lambda v, k: dict(key=(v, k), path=f"{v}_{k}.jpg", t=float(k))
    return [dict(qid=q, video=v, question="q", options=list("abcde"), answer="E", frames=[fr(v, k) for k in range(4)])
            for q, v in ((1, "a"), (2, "a"), (3, "b"))]


def test_run_questions_drops_empty_frames_and_resumes(tmp_path):
    out = str(tmp_path / "dtd.jsonl")
    conds = {"abs 0.5": dict(threshold=0.5, absolute=True), "rel 0.5": dict(threshold=0.5, absolute=False)}
    be = FakeBackend()
    dtd.run_questions(be, _items(), conds, out, build_messages=None)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 6 and all(r["correct"] == 1 and r["error"] is None for r in rows)
    assert sorted(set(be.encoded)) == sorted({(v, k) for v in "ab" for k in range(4)})
    a = next(r for r in rows if r["condition"] == "abs 0.5")                  # identical frames: all dropped
    assert a["n_frames"] == 1 and a["n_frames_dropped"] == 3 and a["n_image_tokens"] == 8 and a["kept_share"] == 0.25
    r = next(r for r in rows if r["condition"] == "rel 0.5")                  # half of each later frame kept
    assert r["n_frames"] == 4 and r["n_image_tokens"] == 8 + 3 * 4 and r["kept_share"] == 0.625
    assert r["dtd"] == dict(threshold=0.5, absolute=False)
    dtd.run_questions(FakeBackend(), _items(), conds, out, build_messages=None)
    assert len(open(out).read().splitlines()) == 6
