"""The frame budget (vlmeval/budget_rule.py) and its rows in accuracy_tables/. CPU only, no data, no model."""
import numpy as np
import pandas as pd
import pytest

from accuracy_tables import common, compare_budget, run
from eyehead import policy as pol
from vlmeval import budget_rule as br

T, P, R, S = pol.T, pol.P, pol.R, pol.S


def labels(n, **pairs):
    """n pairs, STABLE unless named: labels(20, T=range(6, 13), R=[14, 16])."""
    lab = np.full(n, S)
    for name, where in pairs.items():
        lab[list(where)] = dict(T=T, P=P, R=R, S=S)[name]
    return lab


# ---------------------------------------------------------------------------- the frame limit
def test_frame_limit_worked_examples():
    assert br.frame_limit(0.1, 96 * 256, 48, 1100) == 51                     # Qwen-sized numbers from the runbook
    assert [br.frame_limit(b, 64 * 81, 17, 200, 64) for b in (0.05, 0.1, 0.2)] == [15, 30, 60]


def test_frame_limit_stays_within_the_video_and_the_cap():
    assert br.frame_limit(1.0, 96 * 256, 17, 40) == 40
    assert br.frame_limit(1.0, 96 * 256, 17, 500, 64) == 64
    assert br.frame_limit(0.0001, 100, 17, 40) == 1


# ---------------------------------------------------------------------------- the send reasons
def test_reasons_on_a_known_sequence():
    reasons, motion = br.send_reasons(np.array([S, T, P, P, T, S, S, P, P, S]))
    sent = {t: r for t, r in enumerate(reasons) if r}
    assert sent == {0: br.FRAME0, 3: br.PURSUIT, 6: br.SETTLE, 8: br.PURSUIT, 10: br.SETTLE}
    assert motion[6] == 4 and motion[10] == 2 and motion[3] == 0


@pytest.mark.parametrize("seed", range(20))
def test_reasons_send_the_same_frames_as_the_policy(seed):
    lab = np.random.default_rng(seed).integers(0, 4, 300)
    reasons, _ = br.send_reasons(lab)
    mask = pol.send_settle(np.isin(lab, pol.MOVING), lab == R, refix=True, pu=lab == P, pursuit="first")
    assert [r is not None for r in reasons] == mask.tolist()


# ---------------------------------------------------------------------------- the slots
@pytest.mark.parametrize("n,limit", [(1100, 51), (67, 30), (64, 64), (10, 1), (259, 60), (5, 2)])
def test_slots_cover_the_video_evenly(n, limit):
    e = br.slot_edges(n, limit)
    sizes = np.diff(e)
    assert e[0] == 0 and e[-1] == n and len(sizes) == limit
    assert sizes.min() >= 1 and sizes.max() - sizes.min() <= 1


def test_slots_reject_a_limit_above_the_frames():
    with pytest.raises(ValueError):
        br.slot_edges(3, 4)


def test_slot_middles_are_one_per_slot():
    assert br.slot_middles(21, 3) == [3, 10, 17]
    assert br.slot_middles(4, 1) == [1]                                      # two middles: the earlier


# ---------------------------------------------------------------------------- the rule
def test_marks_within_the_limit_are_all_sent_and_nothing_is_added():
    lab = labels(20, R=[4, 9])                                               # marks: 0, 5, 10
    assert br.keep_frames(lab, 3) == ([0, 5, 10], [br.FRAME0, br.REFIX, br.REFIX])
    assert br.keep_frames(lab, 5) == ([0, 5, 10], [br.FRAME0, br.REFIX, br.REFIX])


def test_a_slot_prefers_settle_then_refix_then_pursuit_then_frame0():
    # 3 slots of 7 frames: 0-6, 7-13, 14-20. Slot 0: only frame 0. Slot 1: a refix, then a pursuit start whose moving
    # stretch runs on into slot 2, so slot 1 holds no settle. Slot 2: a pursuit start, then a settle.
    lab = labels(20, R=[7], P=[12, 13, 15], T=[14])
    reasons, _ = br.send_reasons(lab)
    assert [(t, r) for t, r in enumerate(reasons) if r] == [(0, br.FRAME0), (8, br.REFIX), (13, br.PURSUIT),
                                                            (16, br.PURSUIT), (17, br.SETTLE)]
    assert br.keep_frames(lab, 3) == ([0, 8, 17], [br.FRAME0, br.REFIX, br.SETTLE])


def test_a_slot_keeps_a_pursuit_frame_over_frame0():
    lab = labels(20, P=[2], R=[7, 9, 16, 18])                                # slot 0: frame 0, pursuit 3, settle 4
    assert br.keep_frames(lab, 3)[0][0] == 4
    lab = labels(20, P=range(2, 9), R=[9, 11, 16, 18])                       # slot 0: frame 0 and pursuit 3 only
    assert br.keep_frames(lab, 3)[0][0] == 3


def test_the_settle_after_the_longest_movement_wins_then_the_earliest():
    # With a limit of 2, slot 0 is frames 0-10: settles at 3 (after 1 moving pair) and 8 (after 3). Frame 13 is in slot 1.
    lab = labels(20, T=[1, 4, 5, 6, 9, 10, 11])
    reasons, motion = br.send_reasons(lab)
    assert [t for t, r in enumerate(reasons) if r == br.SETTLE] == [3, 8, 13]
    assert [motion[t] for t in (3, 8, 13)] == [1, 3, 3]
    assert br.keep_frames(lab, 2)[0][0] == 8
    lab = labels(20, T=[1, 2, 3, 6, 7, 8, 12, 13])                           # settles at 5 and 10 tie: the earlier
    assert br.keep_frames(lab, 2)[0][0] == 5


def test_an_empty_slot_takes_the_still_frame_nearest_its_middle():
    # Slots 0-6, 7-13, 14-20. Slot 1 has no mark and every frame is still: its middle, frame 10.
    assert br.keep_frames(labels(20, R=[14, 16, 18]), 3) == ([0, 10, 15], [br.FRAME0, br.FILL, br.REFIX])
    # Frames 10 to 14 are moving and the stretch ends in slot 2, so slot 1's nearest still frame is 9.
    lab = labels(20, T=range(9, 14), R=[16, 18])
    assert br.keep_frames(lab, 3) == ([0, 9, 15], [br.FRAME0, br.FILL, br.SETTLE])


def test_an_empty_slot_with_no_still_frame_takes_its_middle():
    lab = labels(20, T=range(6, 13), R=[14, 16, 18])                         # frames 7 to 13 all moving
    assert br.keep_frames(lab, 3) == ([0, 10, 14], [br.FRAME0, br.FILL, br.SETTLE])


@pytest.mark.parametrize("seed", range(20))
def test_one_frame_per_slot_when_over_the_limit(seed):
    lab = np.random.default_rng(seed).integers(0, 4, 400)
    marks = sum(r is not None for r in br.send_reasons(lab)[0])
    limit = marks // 3
    kept, why = br.keep_frames(lab, limit)
    e = br.slot_edges(len(lab) + 1, limit)
    assert len(kept) == len(why) == limit and all(e[i] <= t < e[i + 1] for i, t in enumerate(kept))


# ---------------------------------------------------------------------------- the rows
class FakeAdapter:
    key, stock_tokens = "smolvlm", 81

    def fovea_settings(self, ours):
        return [dict(k=int(k)) for k in ours["smolvlm_k"]]

    def setting_label(self, s):
        return f"k={s['k']}"

    def fovea_condition(self, s):
        return ("fovea", dict(k=s["k"], gaze="gaze"))

    def stock_condition(self):
        return ("uniform", 384)

    def score(self, items, conditions, out_path):
        raise NotImplementedError


def fake_data(n_frames=200, clips=1):
    lab = np.random.default_rng(0).integers(0, 4, n_frames - 1)
    qs = pd.DataFrame(dict(qid=[1, 2], video=["v", "v"], question=["q1", "q2"], options=[list("abcde")] * 2,
                           answer=["A", "B"]))
    clip = dict(sequence="v", frame_no=np.arange(n_frames))
    return dict(qs=qs, clips=[clip] * clips, labels={"v": lab}, metas={"v": dict(n=n_frames)},
                uniform={"v": list(range(96))}, frame=lambda v, f: dict(key=(v, f)))


def config():
    return common._merge(common.DEFAULTS, {})


def test_extra_rows_are_not_in_the_default_rows():
    assert len(run.ROWS) == 11 and not set(run.EXTRA_ROWS) & set(run.ROWS)


def test_row_jobs_gives_one_job_per_budget_on_the_smallest_fovea():
    jobs = run.row_jobs(FakeAdapter(), config(), fake_data(), ("budget", "nogate"))
    assert [j["name"] for j in jobs] == ["budget 5%", "budget 10%", "budget 20%", "nogate 5%", "nogate 10%", "nogate 20%"]
    assert {j["row"] for j in jobs} == {"budget", "nogate"}
    for j in jobs:
        (name, cond), = j["conds"].items()
        assert "k=1" in name and cond == ("fovea", dict(k=1, gaze="gaze"))
    assert [len(j["items"][0]["frames"]) for j in jobs[3:]] == [15, 30, 60]       # no gate always sends the limit
    assert all(len(j["items"][0]["frames"]) <= f for j, f in zip(jobs[:3], (15, 30, 60)))
    assert len({n for j in jobs for n in j["conds"]}) == 6


def test_budget_frames_only_uses_frames_the_video_has():
    data = fake_data()
    data["metas"]["v"]["n"] = 150                                                  # the frame folder is shorter
    frames, why = common.budget_frames(config(), data, "smolvlm", 81, 0.2, use_gate=False)
    assert max(f["key"][1] for f in frames["v"]) < 150 and why == {"middle": 60}


def test_budget_frames_stops_on_a_model_without_a_token_count():
    with pytest.raises(KeyError, match="budget.tokens_per_frame.phi"):
        common.budget_frames(config(), fake_data(), "phi", 757, 0.1, use_gate=True)


def test_budget_frames_stops_on_a_video_split_into_clips():
    with pytest.raises(ValueError, match="more than one clip"):
        common.budget_frames(config(), fake_data(clips=2), "smolvlm", 81, 0.1, use_gate=True)


# ---------------------------------------------------------------------------- the comparison
def test_compare_pairs_each_budget_and_only_calls_a_clear_difference():
    rows = []
    for i in range(40):
        v = f"v{i % 4}"
        common_fields = dict(qid=i, video=v, n_frames=30)
        rows.append(dict(common_fields, row="uniform", condition="uniform K=64", correct=1, n_image_tokens=5184))
        rows.append(dict(common_fields, row="budget", condition="budget rule 10% (D+P gate, fovea k=1)", correct=1,
                         n_image_tokens=510))
        rows.append(dict(common_fields, row="nogate", condition="no gate 10% (slot middles, fovea k=1)", correct=0,
                         n_image_tokens=510))
        rows.append(dict(common_fields, row="budget", condition="budget rule 20% (D+P gate, fovea k=1)", correct=i % 2,
                         n_image_tokens=1020))
        rows.append(dict(common_fields, row="nogate", condition="no gate 20% (slot middles, fovea k=1)", correct=i % 2,
                         n_image_tokens=1020))
    t = compare_budget.compare(pd.DataFrame(rows), B=50).set_index("budget")
    assert list(t.index) == [10.0, 20.0] and t.loc[10.0, "n"] == 40
    assert t.loc[10.0, "rule_minus_nogate"] == pytest.approx(100) and t.loc[10.0, "outcome"] == "rule ahead"
    assert t.loc[20.0, "rule_minus_nogate"] == 0 and t.loc[20.0, "outcome"] == "no clear difference"
    assert t.loc[10.0, "rule_budget_pct"] == pytest.approx(100 * 510 / 5184)
