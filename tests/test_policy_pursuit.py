"""PURSUIT options of policy D (eyehead.policy.send_settle)."""
import numpy as np

from eyehead import policy as pol

T, P, R, S = pol.T, pol.P, pol.R, pol.S
#        pair: 0  1  2  3  4  5  6  7  8  9
LAB = np.array([S, T, P, P, T, S, S, P, P, S])


def sent_frames(pursuit):
    m = pol.send_settle(np.isin(LAB, pol.MOVING), LAB == R, refix=True, pu=LAB == P, pursuit=pursuit)
    return np.nonzero(m)[0].tolist()            # frame t = pair t-1 + 1


def test_default_d_sends_no_pursuit():
    assert sent_frames(None) == [0, 6, 10]           # frame 0, then the settle frames


def test_first_pursuit_frame_of_each_run():
    assert sent_frames("first") == [0, 3, 6, 8, 10]   # pairs 2 and 7 start PURSUIT runs


def test_all_pursuit_frames():
    assert sent_frames("all") == [0, 3, 4, 6, 8, 9, 10]


def test_pursuit_every_n_frames():
    assert sent_frames(2) == [0, 3, 6, 8, 10]


def test_settle_still_sent_after_pursuit():
    # a pursuit send does not answer the moving stretch: frame 6 (settle) is still sent
    assert 6 in sent_frames("first")
