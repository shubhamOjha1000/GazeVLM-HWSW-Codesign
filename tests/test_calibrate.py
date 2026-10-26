"""Budget matching (accuracy_tables/calibrate.py). CPU only, no models, no data.

  1. budgets weighted by questions, as summarize.py measures them
  2. knobs exact from frame counts: StreamingTOM, DyCoke, DTD, average pooling, two-view GazeVLM, each
     checked against the method's own token rule
  3. GazeVLM: the sorted-once heatmap gives exactly gazevlm.mass_box
  4. ours' measured budgets from result files; the matched config
"""
import json

import numpy as np
import pytest

from accuracy_tables import calibrate as cal
from vlmeval import avgpool as ap
from vlmeval import dtd
from vlmeval import dycoke as dy
from vlmeval import gazevlm as gz


def test_weighted_and_closest():
    assert cal.weighted({"a": 10, "b": 30}, {"a": 100, "b": 100}, {"a": 3, "b": 1}) == pytest.approx(60 / 400)
    assert cal.closest([(0.04, "x"), (0.06, "y"), (0.051, "z")], 0.05) == (0.051, "z")


def test_streamingtom_keeps_G_per_frame():
    pred, ratio = cal.match_streamingtom(0.1, {"a": 10}, {"a": 1}, {"a": 2560}, 256)
    assert ratio == 26 / 256 and pred == pytest.approx(26 / 256)                # 26 beats 25 for 25.6 tokens


def test_dycoke_prune_gives_the_intended_keep():
    pred, prune = cal.match_dycoke(0.05, {"a": 64}, {"a": 1}, {"a": 64 * 256}, 256, 32)
    assert dy.keep_count(256, prune) == 5                                        # 2 whole frames + 62 x 5
    assert pred == pytest.approx((2 * 256 + 62 * 5) / (64 * 256))


def test_dtd_threshold_matches_its_retention():
    pred, t = cal.match_dtd(0.053, {"a": 96}, {"a": 1}, {"a": 96 * 256}, 256)
    assert 256 - int(256 * t) == 11 and pred == pytest.approx(dtd.retention(t, 96, 256))


def test_avgpool_and_gazevlm2_knobs_map_back_exactly():
    pred, b = cal.match_avgpool(0.1, 256, 16)
    assert ap.pool_side(b, 256, 16) == 5 and pred == pytest.approx(25 / 256)
    pred, b = cal.match_avgpool(0.05, 757, 24, row_breaks=True)                   # Phi: s^2 + s of 757
    assert ap.pool_side(b, 757, 24, True) == 6 and pred == pytest.approx(42 / 757)
    pred, b = cal.match_gazevlm2(0.05, 256, 256)
    assert max(2, int(round(b * 256))) == 12 and pred == pytest.approx(12 / 256)


def test_prefix_box_is_mass_box():
    rng = np.random.default_rng(0)
    g = dict(gaze="fixation", sigma=0.1, r_min=0.125, thumb=2)
    for u, v in rng.random((5, 2)):
        fr = dict(gaze=(float(u), float(v)))
        pre = cal.heat_prefix(fr, g)
        G = gz.gaze_heatmap([fr["gaze"]], sigma=0.1)
        for rho in (0.01, 0.2, 0.5, 0.9, 1.0):
            assert cal.prefix_box(pre, rho) == gz.mass_box(G, rho)


def test_gazevlm_tokens_grow_with_rho():
    g = dict(gaze="fixation", sigma=0.1, r_min=0.125, thumb=2)
    pres = [cal.heat_prefix(dict(gaze=(0.5, 0.5)), g)]
    toks = [cal.gazevlm_tokens(pres, r, g, 16) for r in cal.GRID_RHO]
    assert toks == sorted(toks) and 4 + 2 * 2 <= toks[0] <= 4 + 3 * 3            # r_min 0.125: 2-3 tokens a side
    fr = dict(gaze=(0.5, 0.5))
    r0, r1, c0, c1 = gz.box_on_grid(gz.roi_box([fr["gaze"]], 0.5, 0.1, 0.125), 16)
    assert cal.gazevlm_tokens(pres, 0.5, g, 16) == 4 + (r1 - r0) * (c1 - c0)    # as gazevlm.run_questions counts


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_ours_budgets_from_result_files(tmp_path):
    d = tmp_path / "qwen2.5"
    d.mkdir()
    base = dict(video="v", answer="A", pred="A", correct=1, error=None)
    _write(d / "uniform.jsonl", [dict(base, condition="uniform K=96 (stock input)", qid=q, n_image_tokens=1000)
                                 for q in (1, 2)])
    _write(d / "ours.jsonl", [dict(base, condition=f"ours D+P gate + fovea {s}", qid=q, n_image_tokens=t)
                              for s, t in (("R=70 a=160", 90), ("R=180 a=40", 200)) for q in (1, 2)]
           + [dict(base, condition="ours D+P gate + fovea R=70 a=160", qid=3, n_image_tokens=0, error="oom")])
    ours = cal.ours_budgets(str(tmp_path), "qwen2.5")
    assert ours == {"ours D+P gate + fovea R=70 a=160": pytest.approx(0.09),
                    "ours D+P gate + fovea R=180 a=40": pytest.approx(0.2)}


def test_matched_config_sets_every_baseline():
    knobs = {m: [dict(target=0.09, knob=k, predicted=0.09)] for m, k in
             (("streamingtom", 23 / 256), ("dycoke", 0.95), ("dtd", 0.9), ("avgpool", 25 / 256), ("gazevlm2", 22 / 256),
              ("gazevlm", 0.3), ("prunevid", 0.12), ("fastvid", 0.085))}
    cfg = dict(dycoke=dict(settings=[dict(window=32, prune=0.98)]), prunevid=dict(settings=[dict(tau=0.8, beta=0.07)]))
    out = cal.matched_config(dict(paths=dict(data_root="/x"), streamingtom=dict(tau=0.9)), knobs, cfg)
    assert out["streamingtom"] == dict(tau=0.9, ratio=[23 / 256]) and out["paths"] == dict(data_root="/x")
    assert out["dycoke"]["settings"] == [dict(window=32, prune=0.95)]
    assert out["dtd"]["settings"] == [dict(threshold=0.9, absolute=False)]
    assert out["prunevid"]["settings"] == [dict(tau=0.8, beta=0.12)] and out["fastvid"]["r"] == [0.085]
    assert out["gazevlm"]["rho"] == [0.3] and out["avgpool"]["budget"] == [25 / 256] and out["gazevlm2"]["budget"] == [22 / 256]
