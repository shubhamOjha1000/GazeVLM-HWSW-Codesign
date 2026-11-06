"""Budget matching: set every baseline's knob so that its token budget equals ours' measured budget.

    python -m accuracy_tables.calibrate --config accuracy_tables/config.yaml --model qwen2.5 --points 2 \
        --write accuracy_tables/config_qwen2.5_matched.yaml

1. Ours' budget per fovea setting, measured: mean visual tokens per question / the Uniform K row's on the
   same questions, from <out_dir>/<model>/{ours,uniform}.jsonl (run --rows uniform,ours first). The
   matched budgets are ours' `--points` smallest (or the settings named with `--match`).
2. For each matched budget b, each baseline's knob is chosen so that its budget on the same questions (the
   Uniform K frames) is closest to b, computed the way summarize.py measures it (per-video token counts
   weighted by questions):
     exact from the frame counts   StreamingTOM G / N, DyCoke's pruning rate K, DTD's drop share,
                                   average pooling's s x s, the two-view GazeVLM's tokens per frame
     from the gaze                 GazeVLM's rho (ROI + thumbnail tokens per frame)
     from the frames' tokens       PruneVid's beta and FastVID's r: the methods' own plans on the encoded
                                   frames (the model's encoder only, no LLM)
   Only token counts are used, never answers, so nothing is tuned on the test questions' accuracy.
3. Writes a copy of the config with those knobs (one condition per matched budget per baseline) and
   <out_dir>/<model>/calibration.json (ours' budgets, targets, knobs, predicted budgets).

With --fractions, step 1 is skipped: the targets are the config's budget.fractions (5, 10 and 20%), so nothing needs
scoring first, and it also writes <out_dir>/<model>/targets.csv, which tells the report each row's condition at each budget.
Not on Phi yet: its baselines add row breaks that these counts leave out (runbook section 11).
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accuracy_tables import common, summarize  # noqa: E402
from accuracy_tables.models import ADAPTERS  # noqa: E402

GRID_BETA = [round(0.01 * i, 2) for i in range(1, 61)]          # PruneVid cluster ratios tried
GRID_R = [round(0.005 * i, 3) for i in range(2, 101)]           # FastVID retention ratios tried
GRID_RHO = [round(0.01 * i, 2) for i in range(1, 96)]           # GazeVLM gaze-mass fractions tried
UNCOUNTED_ROW_BREAKS = ("phi",)                                 # models whose baselines' row breaks these counts leave out
NEAREST_ONLY_PTS = 0.5                                          # further than this from its target, a row only reached the nearest


# ---------------------------------------------------------------------------- budgets as summarize.py measures them
def weighted(kept, full, nq):
    """Budget (fraction): questions x kept tokens over questions x Uniform K tokens, per video."""
    return sum(nq[v] * kept[v] for v in nq) / sum(nq[v] * full[v] for v in nq)


def closest(cands, target):
    """cands: [(predicted budget, knob)] -> the one whose budget is closest to target (first on a tie)."""
    return min(cands, key=lambda c: abs(c[0] - target))


# ---------------------------------------------------------------------------- knobs exact from frame counts
def match_streamingtom(b, n, nq, full, N):
    """G tokens kept in every frame (stm.budget(ratio, N) = round(ratio N)) -> ratio = G / N."""
    return closest([(weighted({v: n[v] * G for v in n}, full, nq), G / N) for G in range(1, N + 1)], b)


def match_dycoke(b, n, nq, full, N, window):
    """Each window's first frame whole, every other frame keeps k = keep_count(N, K) tokens; K is chosen
    halfway between two integer k so that int((1 - K) N) = k exactly."""
    first = {v: math.ceil(n[v] / window) for v in n}
    cands = [(weighted({v: first[v] * N + (n[v] - first[v]) * k for v in n}, full, nq), 1 - (k + 0.5) / N)
             for k in range(1, N + 1)]
    return closest(cands, b)


def match_dtd(b, n, nq, full, N):
    """Relative DTD: the first frame whole, every later frame drops int(t N) -> keeps k; t halfway so that
    int(t N) = N - k exactly."""
    cands = [(weighted({v: N + (n[v] - 1) * k for v in n}, full, nq), (N - k + 0.5) / N) for k in range(1, N + 1)]
    return closest(cands, b)


def match_avgpool(b, stock, side, row_breaks=False):
    """s x s tokens (+ s row breaks on Phi) in every frame -> the budget share that run.py maps to that s."""
    from vlmeval import avgpool as ap
    cands = [(ap.tokens_per_frame(s, row_breaks) / stock, ap.tokens_per_frame(s, row_breaks) / stock)
             for s in range(1, side + 1)]
    return closest(cands, b)


def match_gazevlm2(b, stock, n_tokens):
    """B tokens per frame, 2 (B // 2) after the halving per view -> the budget share run.py maps to B."""
    cands = []
    for B in range(2, 2 * n_tokens + 1, 2):
        cands.append((B / stock, B / stock))
    return closest(cands, b)


# ---------------------------------------------------------------------------- knobs from the gaze / the tokens
def heat_prefix(fr, g):
    """A frame's gaze heatmap sorted once as gazevlm.mass_box sorts it: cumulative mass and the running
    bounding box of the first k cells, so that the rho-mass box of every rho is a lookup."""
    from vlmeval import gazevlm as gz
    pts, _ = gz.gaze_points(fr, g["gaze"])
    G = gz.gaze_heatmap(pts, sigma=float(g["sigma"]))
    flat = G.ravel()
    order = np.argsort(-flat, kind="stable")
    ys, xs = np.unravel_index(order, G.shape)
    return (G.shape[0], np.cumsum(flat[order]), np.minimum.accumulate(xs), np.minimum.accumulate(ys),
            np.maximum.accumulate(xs), np.maximum.accumulate(ys))


def prefix_box(pre, rho):
    """gazevlm.mass_box(G, rho) from heat_prefix(...)."""
    n, cum, x0, y0, x1, y1 = pre
    k = min(int(np.searchsorted(cum, rho)) + 1, len(cum)) - 1
    return x0[k] / n, y0[k] / n, (x1[k] + 1) / n, (y1[k] + 1) / n


def gazevlm_tokens(prefixes, rho, g, side):
    """GazeVLM tokens of a video's frames at rho: thumb^2 + the ROI's token rows x columns, per frame."""
    from vlmeval import gazevlm as gz
    total = 0
    for pre in prefixes:
        r0, r1, c0, c1 = gz.box_on_grid(gz.grow_box(prefix_box(pre, rho), float(g["r_min"])), side)
        total += int(g["thumb"]) ** 2 + (r1 - r0) * (c1 - c0)
    return total


def prunevid_kept(states, beta, s):
    from vlmeval import prunevid as pv
    plans, _, _ = pv.prunevid_plans(states, float(s["gamma"]), float(s["settings"][0]["tau"]), beta,
                                    int(s["knn"]), int(s["min_tokens"]))
    return sum(len(rows) for rows in plans)


def fastvid_kept(states, r, s):
    from vlmeval import fastvid as fv
    plans, _ = fv.fastvid_plans(states, r, int(s["c"]), float(s["tau"]), float(s["ignore"]), float(s["d"]),
                                int(s["p"]), float(s["beta"]))
    return sum(len(kept) for kept, _ in plans)


# ---------------------------------------------------------------------------- the whole calibration
def ours_budgets(out_dir, key):
    """{ours condition: measured budget (fraction)} from the ours and uniform result files."""
    res = summarize.load_model_results(out_dir, key)
    if not len(res) or not (res.row == "uniform").any() or not (res.row == "ours").any():
        raise SystemExit(f"{out_dir}/{key}: score --rows uniform,ours first (no results to measure)")
    bud = summarize.budgets(res)
    conds = sorted(set(res[res.row == "ours"].condition))
    return {c: bud[c] / 100 for c in conds if np.isfinite(bud.get(c, np.nan))}


def condition_name(method, knob, cfg, stock, pool_grid, k):
    """The name run.py gives a baseline's condition at this knob. pool_grid: (side, row breaks) of average pooling."""
    from accuracy_tables import run
    from vlmeval import avgpool as ap
    if method == "avgpool":
        return run.avgpool_name(ap.pool_side(float(knob), stock, *pool_grid), k)
    if method == "gazevlm2":
        return run.gazevlm2_name(max(2, int(round(float(knob) * stock))), cfg["gazevlm2"], k)
    if method == "gazevlm":
        return run.gazevlm_name(knob, cfg["gazevlm"], k)
    if method == "streamingtom":
        return run.streamingtom_name(knob, cfg["streamingtom"], k)
    if method == "dycoke":
        return run.dycoke_name(cfg["dycoke"]["settings"][0]["window"], knob, k)
    if method == "dtd":
        return run.dtd_name(dict(threshold=knob, absolute=False), k)
    if method == "fastvid":
        return run.fastvid_name(knob, cfg["fastvid"], k)
    if method == "prunevid":
        return run.prunevid_name(dict(tau=cfg["prunevid"]["settings"][0]["tau"], beta=knob), cfg["prunevid"], k)
    raise ValueError(f"no condition name for {method}")


def targets_table(key, knobs, cfg):
    """One line per row and budget: the condition a run must score and the report must find. Two budgets can share a
    condition when a row can't go lower, and then each has its own line."""
    import pandas as pd
    from accuracy_tables import run
    ad = ADAPTERS[key]()
    cap = cfg["run"]["max_frames"].get(key)
    k = min(cfg["run"]["K"], cap) if cap else cfg["run"]["K"]
    label = ad.setting_label(ad.fovea_settings(cfg["ours"])[0])
    lines = [dict(model=key, row="uniform", target_pct=100.0, condition=run.uniform_name(k)),
             dict(model=key, row="blind", target_pct=0.0, condition=run.BLIND)]
    lines += [dict(model=key, row="budget", target_pct=100 * float(f), condition=run.budget_name(f, cfg["run"]["policy"], label))
              for f in cfg["budget"]["fractions"]]
    lines += [dict(model=key, row=m, target_pct=100 * x["target"], knob=x["knob"], predicted_pct=100 * x["predicted"],
                   condition=x["condition"]) for m, ks in knobs.items() for x in ks]
    return pd.DataFrame(lines)


def calibrate(cfg, key, targets, log=print):
    """-> {method: [{target, knob, predicted, condition}]} for the matched budgets (fractions)."""
    data = common.load_data(cfg)
    ad = ADAPTERS[key](cfg["models"].get(key), cfg).load()
    try:
        cap = cfg["run"]["max_frames"].get(key)
        k = min(cfg["run"]["K"], cap) if cap else cfg["run"]["K"]
        frames = {v: common.cap_frames(f, cap) for v, f in data["uniform"].items()}
        nq = data["qs"].video.value_counts().to_dict()
        frames = {v: frames[v] for v in nq}
        n = {v: len(frames[v]) for v in frames}
        stock = ad.stock_tokens
        full = {v: n[v] * stock for v in n}
        gz_be = ad.gazevlm_backend()
        side = gz_be.side
        N = side * side                                              # tokens of a frame's grid (256 on Qwen2.5)
        ap_be = ad.avgpool_backend()
        out = {m: [] for m in ("streamingtom", "dycoke", "dtd", "avgpool", "gazevlm2", "gazevlm", "prunevid", "fastvid")}
        window = int(cfg["dycoke"]["settings"][0]["window"])
        g = cfg["gazevlm"]
        kept_gz = {r: {} for r in GRID_RHO}
        for v in frames:                     # one video at a time: a frame's sorted heatmap takes about 2 MB
            prefixes = [heat_prefix(fr, g) for fr in frames[v]]
            for r in GRID_RHO:
                kept_gz[r][v] = gazevlm_tokens(prefixes, r, g, side)
        gz_cands = [(weighted(kept_gz[r], full, nq), r) for r in GRID_RHO]
        for b in targets:
            for m, (pred, knob) in (("streamingtom", match_streamingtom(b, n, nq, full, N)),
                                    ("dycoke", match_dycoke(b, n, nq, full, N, window)),
                                    ("dtd", match_dtd(b, n, nq, full, N)),
                                    ("avgpool", match_avgpool(b, stock, ap_be.side, getattr(ap_be, "row_breaks", False))),
                                    ("gazevlm2", match_gazevlm2(b, stock, N)),
                                    ("gazevlm", closest(gz_cands, b))):
                out[m].append(dict(target=b, knob=knob, predicted=pred))
        log("encoding the Uniform K frames (encoder only) for PruneVid's and FastVID's plans ...")
        fv_be = ad.fastvid_backend()
        kept_pv = {beta: {} for beta in GRID_BETA}
        kept_fv = {r: {} for r in GRID_R}
        workers = int(cfg["run"]["prefetch_workers"])
        with torch.no_grad():
            for v in frames:
                if workers:
                    fv_be.prefetch(frames[v], workers)
                states = [fv_be.frame_state(fr) for fr in frames[v]]
                for beta in GRID_BETA:
                    kept_pv[beta][v] = prunevid_kept(states, beta, cfg["prunevid"])
                for r in GRID_R:
                    kept_fv[r][v] = fastvid_kept(states, r, cfg["fastvid"])
                log(f"   {v}: {len(states)} frames")
                del states
                torch.cuda.empty_cache()
        for b in targets:
            pred, knob = closest([(weighted(kept_pv[beta], full, nq), beta) for beta in GRID_BETA], b)
            out["prunevid"].append(dict(target=b, knob=knob, predicted=pred))
            pred, knob = closest([(weighted(kept_fv[r], full, nq), r) for r in GRID_R], b)
            out["fastvid"].append(dict(target=b, knob=knob, predicted=pred))
        pool_grid = (ap_be.side, getattr(ap_be, "row_breaks", False))
        for m, ks in out.items():
            for x in ks:
                x["condition"] = condition_name(m, x["knob"], cfg, stock, pool_grid, k)
        return out
    finally:
        ad.free()


def matched_config(raw, knobs, cfg):
    """The raw (unresolved) config with every baseline's section set to the matched knobs."""
    import copy
    raw = copy.deepcopy(raw)
    ks = lambda m: [k["knob"] for k in knobs[m]]
    st = raw.setdefault("streamingtom", {})
    st["ratio"] = ks("streamingtom")
    window = int(cfg["dycoke"]["settings"][0]["window"])
    raw.setdefault("dycoke", {})["settings"] = [dict(window=window, prune=k) for k in ks("dycoke")]
    raw.setdefault("dtd", {})["settings"] = [dict(threshold=k, absolute=False) for k in ks("dtd")]
    raw.setdefault("avgpool", {})["budget"] = ks("avgpool")
    raw.setdefault("gazevlm2", {})["budget"] = ks("gazevlm2")
    raw.setdefault("gazevlm", {})["rho"] = ks("gazevlm")
    tau = float(cfg["prunevid"]["settings"][0]["tau"])
    raw.setdefault("prunevid", {})["settings"] = [dict(tau=tau, beta=k) for k in ks("prunevid")]
    raw.setdefault("fastvid", {})["r"] = ks("fastvid")
    return raw


def main(argv=None):
    import yaml
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="qwen2.5", help=f"one of {', '.join(ADAPTERS)}")
    ap.add_argument("--points", type=int, default=2, help="match ours' N smallest measured budgets (default 2)")
    ap.add_argument("--match", help="comma list of substrings naming the ours settings to match instead")
    ap.add_argument("--fractions", action="store_true",
                    help="match the config's budget.fractions instead of ours' measured budgets; nothing needs scoring first")
    ap.add_argument("--videos", help="override run.videos: test, all, or a comma list of video names")
    ap.add_argument("--out-dir", help="override paths.out_dir (where ours and uniform were scored)")
    ap.add_argument("--prefetch-workers", type=int, help="override run.prefetch_workers: threads loading frames ahead while encoding")
    ap.add_argument("--write", required=True, help="the matched config to write (next to --config keeps relative paths)")
    a = ap.parse_args(argv)
    if a.fractions and a.model in UNCOUNTED_ROW_BREAKS:
        sys.exit(f"{a.model}: its baselines add row breaks that these counts leave out, so --fractions would land above "
                 "its targets. The user decides this first: see runbook section 11.")
    if a.prefetch_workers and not ADAPTERS[a.model].baselines_load_on_threads:
        ap.error(f"frame loading threads aren't built for the baselines of {a.model}: leave out --prefetch-workers")

    cfg = common.load_config(a.config)
    if a.prefetch_workers is not None:
        cfg["run"]["prefetch_workers"] = a.prefetch_workers
    if a.videos:
        cfg["run"]["videos"] = a.videos if a.videos in ("test", "all") else [v.strip() for v in a.videos.split(",")]
    if a.out_dir:
        cfg["paths"]["out_dir"] = os.path.abspath(os.path.expanduser(a.out_dir))
    out_dir = cfg["paths"]["out_dir"]

    if a.fractions:
        ours, picked = None, None
        targets = [float(f) for f in cfg["budget"]["fractions"]]
    else:
        ours = ours_budgets(out_dir, a.model)
        print("ours' measured token budgets (vs Uniform K, same questions):")
        for c, b in sorted(ours.items(), key=lambda x: x[1]):
            print(f"   {100 * b:6.2f}%  {c}")
        if a.match:
            picked = [c for c in ours if any(m.strip() in c for m in a.match.split(","))]
        else:
            picked = sorted(ours, key=ours.get)[:a.points]
        if not picked:
            raise SystemExit(f"no ours setting matches {a.match!r}")
        targets = [float(ours[c]) for c in sorted(picked, key=ours.get)]
    print(f"matching every baseline to {[f'{100 * t:.2f}%' for t in targets]}")

    knobs = calibrate(cfg, a.model, targets)
    print("\nbaseline knobs (predicted budget, %):")
    for m, ks in knobs.items():
        print(f"   {m:<13} " + "   ".join(f"{k['knob']:g} ({100 * k['predicted']:.2f}%)" for k in ks))

    raw = yaml.safe_load(open(a.config))
    out = matched_config(raw, knobs, cfg)
    if a.videos:
        out.setdefault("run", {})["videos"] = cfg["run"]["videos"]
    if a.out_dir:
        out.setdefault("paths", {})["out_dir"] = out_dir
    with open(a.write, "w") as f:
        what = "the token budgets" if a.fractions else "ours' measured budgets"
        f.write(f"# Budget-matched copy of {os.path.basename(a.config)}: every baseline set to {what} "
                f"{[round(100 * float(t), 2) for t in targets]} % ({a.model}); written by accuracy_tables/calibrate.py\n")
        yaml.safe_dump(out, f, sort_keys=False)
    rec = dict(model=a.model, ours=ours, matched=picked, targets=targets, knobs=knobs)
    os.makedirs(os.path.join(out_dir, a.model), exist_ok=True)
    with open(os.path.join(out_dir, a.model, "calibration.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"\nwritten {a.write} and {os.path.join(out_dir, a.model, 'calibration.json')}")
    if a.fractions:
        table = targets_table(a.model, knobs, cfg)
        table.to_csv(os.path.join(out_dir, a.model, "targets.csv"), index=False)
        reach = table[table.predicted_pct.notna()].copy()
        reach["miss_pts"] = reach.predicted_pct - reach.target_pct
        reach["nearest_only"] = reach.miss_pts.abs() > NEAREST_ONLY_PTS
        print(f"\npredicted budget per row and target, from token counts only. nearest_only: more than {NEAREST_ONLY_PTS} points off:")
        print(reach[["row", "target_pct", "predicted_pct", "miss_pts", "nearest_only", "knob"]].round(3).to_string(index=False))
        print(f"{reach.condition.nunique()} baseline conditions for {len(reach)} row and budget pairs")
        print(f"TARGETS WRITTEN: {os.path.join(out_dir, a.model, 'targets.csv')}")


if __name__ == "__main__":
    main()
