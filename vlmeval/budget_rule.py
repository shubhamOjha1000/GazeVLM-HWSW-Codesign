"""Fit the gate's frames into a token budget: the frame limit, the slots, and the frame kept in each slot.

The gate (policy D+P, eyehead.policy.send_settle) marks frames from the eyes and head alone. Uniform K stops at K
frames however long the video is, but the marks keep growing with the video, so on a long video a low budget can't pay
for them all. This module cuts them down:

    1. The limit: budget x that video's Uniform K tokens / tokens per frame, rounded down.
    2. At most that many marks: send them all and add nothing, so a quiet video simply uses fewer.
    3. More marks than that: cut the video into `limit` equal slots and keep one frame from each:
           a settle frame (the longest moving stretch before it wins, then the earliest)
           otherwise a refixation frame, otherwise a pursuit frame, otherwise frame 0 (each: the earliest)
           if the slot holds no mark: the frame predicted not moving nearest the slot's middle
           (if every frame in the slot is predicted moving: the middle frame)

The no-gate control uses the same slots, takes each slot's middle frame, never reads the labels and always sends
`limit` frames. Only predicted labels are used here, never true labels or answers.

Frames are positions in a clip: frame t ends pair t - 1, so n pairs give n + 1 frames, and frame 0 has no pair.
"""

from eyehead import policy as pol

FRAME0, SETTLE, REFIX, PURSUIT, FILL, MIDDLE = "frame0", "settle", "refix", "pursuit", "fill", "middle"
_RANK = {SETTLE: 3, REFIX: 2, PURSUIT: 1, FRAME0: 0}


def frame_limit(budget, full_tokens, tokens_per_frame, n_frames, cap=None):
    """Frames the budget pays for, rounded down: at least 1, at most the video's frames and the model's frame cap."""
    limit = max(1, int(budget * full_tokens / tokens_per_frame))
    return min(limit, n_frames, cap) if cap else min(limit, n_frames)


def send_reasons(lab):
    """Why policy D+P sends each frame -> (reasons, motion), both of length n + 1.

    reasons[t] is frame0, settle, refix, pursuit or None (not sent). motion[t] is the length of the moving stretch a
    settle frame ends, else 0. Sends the same frames as send_settle(refix=True, pursuit='first'); the tests check it."""
    lab = [int(L) for L in lab]
    reasons, motion = [None] * (len(lab) + 1), [0] * (len(lab) + 1)
    reasons[0] = FRAME0
    stretch = 0
    for k, L in enumerate(lab):
        t = k + 1
        if L in pol.MOVING:
            stretch += 1
            if L == pol.P and (k == 0 or lab[k - 1] != pol.P):
                reasons[t] = PURSUIT
        elif stretch:
            reasons[t], motion[t], stretch = SETTLE, stretch, 0
        elif L == pol.R:
            reasons[t] = REFIX
    return reasons, motion


def slot_edges(n_frames, limit):
    """limit + 1 edges cutting n_frames into equal slots, sizes differing by at most 1 (integer arithmetic, so exact)."""
    if not 1 <= limit <= n_frames:
        raise ValueError(f"need 1 <= limit <= n_frames, got limit={limit}, n_frames={n_frames}")
    return [(2 * i * n_frames + limit) // (2 * limit) for i in range(limit + 1)]


def slot_middles(n_frames, limit):
    """The no-gate control: the middle frame of each slot (the earlier one if a slot has two middles)."""
    edges = slot_edges(n_frames, limit)
    return [(lo + hi - 1) // 2 for lo, hi in zip(edges, edges[1:])]


def _fill(lab, lo, hi):
    """A slot with no mark: the frame predicted not moving nearest the slot's middle, else the middle frame."""
    middle = (lo + hi - 1) / 2
    still = [t for t in range(lo, hi) if t == 0 or int(lab[t - 1]) not in pol.MOVING]
    return min(still, key=lambda t: (abs(t - middle), t)) if still else (lo + hi - 1) // 2


def keep_frames(lab, limit):
    """The budget rule on one clip's predicted labels -> (frames kept, why each was kept), in time order."""
    reasons, motion = send_reasons(lab)
    marks = [t for t, r in enumerate(reasons) if r]
    if len(marks) <= limit:
        return marks, [reasons[t] for t in marks]
    edges = slot_edges(len(reasons), limit)
    kept, why = [], []
    for lo, hi in zip(edges, edges[1:]):
        inside = [t for t in marks if lo <= t < hi]
        if inside:
            t = max(inside, key=lambda f: (_RANK[reasons[f]], motion[f], -f))
            kept.append(t)
            why.append(reasons[t])
        else:
            kept.append(_fill(lab, lo, hi))
            why.append(FILL)
    return kept, why
