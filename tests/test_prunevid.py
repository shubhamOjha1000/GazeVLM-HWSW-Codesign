"""PruneVid baseline (vlmeval/prunevid.py). CPU only, no models, no data.

  1. DPC-KNN clustering: one centre per blob, every token to its nearest centre
  2. refine_clusters: the authors' rules (longest runs kept, the rest to the longer neighbour, left on ties)
  3. temporal segmentation
  4. static positions: the mean pairwise cosine > tau
  5. spatial merging: int(beta m) clusters averaged, only for sets of more than 14 tokens
  6. plans: static tokens averaged over the segment, with its first frame; dynamic tokens per frame
  7. run_questions: frames left without tokens leave the prompt, records, resumable
"""
import json

import numpy as np
import torch

from vlmeval import prunevid as pv
from vlmeval import streamingtom as stm


def _blobs(sizes, d=16, seed=0, scale=0.01):
    """Rows of blob i all along direction i (plus a little noise)."""
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.eye(d)[i] + scale * torch.randn(d, generator=g) for i, n in enumerate(sizes) for _ in range(n)])


# ---------------------------------------------------------------------------- clustering and segments
def test_cluster_one_centre_per_blob():
    centres, labels = pv.cluster(_blobs([5, 6, 4]), 3, knn=3)
    assert labels.tolist() == [0] * 5 + [1] * 6 + [2] * 4
    assert labels[centres].tolist() == [0, 1, 2]                               # a centre is its own cluster


def test_refine_clusters_follows_the_authors_rules():
    assert pv.refine_clusters([0, 0, 1, 1, 0, 0, 0]) == [1, 1, 1, 1, 0, 0, 0]  # shorter run of 0 -> right neighbour (longer)
    assert pv.refine_clusters([0, 0, 2, 1, 1]) == [0, 0, 0, 1, 1]              # one-frame cluster, tie -> left
    assert pv.refine_clusters([0, 1, 2]) == [0, 0, 0]                          # no neighbours left -> 0
    assert pv.refine_clusters([0, 0, 1, 0, 0]) == [0, 0, 0, 0, 0]              # equal longest runs are both kept
    assert pv.refine_clusters([2, 2, 2, 5, 5]) == [2, 2, 2, 5, 5]


def test_segments():
    assert pv.segments(_blobs([4, 3, 3]), gamma=0.3, knn=3) == [4, 3, 3]       # int(0.3 * 10) = 3 clusters
    assert pv.segments(_blobs([4, 3, 3]), gamma=0.01) == [10]                 # int(0.1) = 0 -> one cluster
    assert pv.segments(_blobs([1])) == [1]


# ---------------------------------------------------------------------------- static tokens and spatial merging
def test_static_positions():
    e = torch.eye(2)
    X = torch.stack([torch.stack([e[0], e[0], torch.tensor([1.0, 0.1])]),
                     torch.stack([e[0], e[1], torch.tensor([1.0, -0.1])]),
                     torch.stack([e[0], e[0], torch.tensor([1.0, 0.0])])])    # (3 frames, 3 positions, 2)
    assert pv.static_positions(X, 0.8).tolist() == [True, False, True]         # position 1: mean cosine 1/3
    assert not pv.static_positions(X[:1], 0.8).any()                           # one frame: nothing static
    R = torch.randn(4, 6, 5, generator=torch.Generator().manual_seed(0))
    Rn = R / R.norm(dim=-1, keepdim=True)
    v = torch.stack([torch.stack([Rn[w, i] @ Rn[t, i] for w in range(4) for t in range(4) if w != t]).mean()
                     for i in range(6)])                                       # mean over ordered pairs w != t
    s = torch.sort(v).values
    tau = float(s[2] + s[3]) / 2                                               # between two values: no float ties
    assert pv.static_positions(R, tau).tolist() == (v > tau).tolist()


def test_spatial_merge():
    x = _blobs([10, 10])
    extra = torch.arange(20.0)[:, None]
    c, (rows, ex) = pv.spatial_merge([x, extra], beta=0.1)                     # 20 > 14 tokens: int(2.0) clusters
    assert len(c) == 2 and c[0] < 10 <= c[1]
    assert torch.allclose(rows, torch.stack([x[:10].mean(0), x[10:].mean(0)]), atol=1e-6)
    assert ex.flatten().tolist() == [4.5, 14.5]
    c, (rows,) = pv.spatial_merge([x[:14]], beta=0.1)                          # 14 tokens: left as they are
    assert c.tolist() == list(range(14)) and torch.equal(rows, x[:14])
    c, (rows,) = pv.spatial_merge([x[:15]], beta=0.01)                         # int(0.15) = 0 -> one cluster
    assert len(c) == 1 and torch.allclose(rows[0], x[:15].mean(0), atol=1e-6)


# ---------------------------------------------------------------------------- plans
def test_plans_static_with_the_first_frame_dynamic_per_frame():
    e = torch.eye(8)
    f0 = torch.stack([e[0] + 0.1 * e[1], e[2], e[3], e[4], e[5], e[6]])
    f1 = torch.stack([e[0] - 0.1 * e[1], e[2], e[3], e[5], e[6], e[7]])        # positions 0-2 static, 3-5 dynamic
    states = [dict(tokens=f, deep=[2 * f]) for f in (f0, f1)]
    plans, sizes, share = pv.prunevid_plans(states)
    assert sizes == [2] and share == 0.5
    assert sorted(plans[0]) == [0, 1, 2, 3, 4, 5] and sorted(plans[1]) == [3, 4, 5]
    assert torch.allclose(plans[0][0][0], e[0]) and torch.allclose(plans[0][0][1], 2 * e[0])   # averaged over time
    assert torch.equal(plans[0][3][0], e[4]) and torch.equal(plans[1][3][0], e[5])


def test_plans_merge_dynamic_tokens_inside_a_frame():
    x = _blobs([10, 10])
    plans, sizes, share = pv.prunevid_plans([dict(tokens=x)], beta=0.1)
    assert sizes == [1] and share == 0.0
    a, b = sorted(plans[0])                                                    # 20 dynamic tokens -> 2 clusters
    assert a < 10 <= b
    assert torch.allclose(plans[0][a][0], x[:10].mean(0), atol=1e-6)


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
    out = str(tmp_path / "prunevid.jsonl")
    conds = {"tau=0.8 beta=0.5": dict(tau=0.8, beta=0.5)}
    be = FakeBackend()
    pv.run_questions(be, _items(), conds, out, build_messages=None)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 3 and all(r["correct"] == 1 and r["error"] is None for r in rows)
    assert sorted(be.encoded) == sorted({(v, k) for v in "ab" for k in range(4)})
    r = rows[0]                                                                # all static: one segment, its first frame
    assert r["n_frames"] == 1 and r["n_frames_dropped"] == 3 and r["n_image_tokens"] == 8
    assert r["n_segments"] == 1 and r["static_share"] == 1.0
    assert r["prunevid"]["beta"] == 0.5 and r["prunevid"]["gamma"] == pv.GAMMA
    pv.run_questions(FakeBackend(), _items(), conds, out, build_messages=None)
    assert len(open(out).read().splitlines()) == 3
