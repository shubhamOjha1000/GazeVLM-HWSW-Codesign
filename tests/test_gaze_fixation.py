"""One gaze point per frame from the ~10 samples before it (vlmeval.spatial.gaze_fixation):
the recency-weighted centroid of the densest fixation cluster, not the plain mean."""
import numpy as np

from vlmeval import spatial as sp

TS = np.arange(10) * 0.1 + 0.05            # 10 samples in the second before t = 1.0
A, B = (0.2, 0.2), (0.7, 0.6)


def jitter(n, seed=0, s=0.002):
    return np.random.default_rng(seed).normal(0, s, (n, 2))


def test_picks_the_cluster_with_most_mass():
    xy = np.array([A] * 3 + [B] * 7) + jitter(10)
    assert np.allclose(sp.gaze_fixation((TS, xy), 1.0), B, atol=0.005)


def test_ignores_a_single_outlier():
    xy = np.array([A] * 9 + [(0.9, 0.1)]) + jitter(10)     # the outlier is the most recent sample
    g = sp.gaze_fixation((TS, xy), 1.0)
    assert np.allclose(g, A, atol=0.005)
    assert np.abs(np.array(g) - xy.mean(0)).max() > 0.05   # the plain mean is pulled away


def test_recency_breaks_a_tie():
    xy = np.array([A] * 5 + [B] * 5) + jitter(10)          # equal counts, B is more recent
    assert np.allclose(sp.gaze_fixation((TS, xy), 1.0), B, atol=0.005)


def test_one_fixation_gives_its_centre():
    xy = np.array([B] * 10) + jitter(10, s=0.004)
    assert np.allclose(sp.gaze_fixation((TS, xy), 1.0), B, atol=0.004)


def test_no_samples_gives_none():
    assert sp.gaze_fixation((TS, np.array([A] * 10)), 5.0) is None
