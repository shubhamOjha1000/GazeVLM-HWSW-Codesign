"""Turn the result files of run.py into the accuracy-table rows.

    python -m accuracy_tables.summarize --config accuracy_tables/config.yaml

Per model and condition: questions scored, accuracy with a 95% interval from resampling videos,
frames and visual tokens per question, and the token budget = visual tokens per question / those
of the model's Uniform K row on the same questions (the tables' "100%"). Writes
<out_dir>/summary.csv and prints the accuracy table (rows x models).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accuracy_tables import common  # noqa: E402
from accuracy_tables.models import ADAPTERS  # noqa: E402

FILES = ("uniform", "blind", "ours", "gazevlm", "streamingtom")


def load_model_results(out_dir, key):
    from vlmeval import egoev
    parts = [egoev.load_results(os.path.join(out_dir, key, f"{f}.jsonl")).assign(row=f) for f in FILES]
    res = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
    if len(res) and "error" in res:
        res = res[res["error"].isna()]
    return res


def budgets(res):
    """Token budget (%) per condition vs the Uniform K row, on the questions both scored."""
    uni = res[res.row == "uniform"]
    if not len(uni):
        return {}
    ref = uni.groupby("qid").n_image_tokens.mean()
    out = {}
    for cond, g in res.groupby("condition"):
        g = g.set_index("qid")
        common_q = g.index.intersection(ref.index)
        out[cond] = (100 * g.loc[common_q, "n_image_tokens"].mean() / ref.loc[common_q].mean()
                     if len(common_q) else np.nan)
    return out


def summarize(cfg, videos=None):
    from vlmeval import egoev
    out_dir = cfg["paths"]["out_dir"]
    rows = []
    for key in ADAPTERS:
        res = load_model_results(out_dir, key)
        if not len(res):
            continue
        if videos is not None:
            res = res[res.video.isin(videos)]
        res = res.copy()
        res["correct"] = res["correct"].astype(int)
        acc = egoev.accuracy_table(res)
        bud = budgets(res)
        for cond, r in acc.iterrows():
            g = res[res.condition == cond]
            rows.append(dict(model=key, row=g.row.iloc[0], condition=cond, n=int(r.n), accuracy=r.accuracy,
                             ci_low=r.ci_low, ci_high=r.ci_high, frames_per_q=r.frames_per_q,
                             tokens_per_q=g.n_image_tokens.mean(), budget_pct=bud.get(cond, np.nan),
                             videos=g.video.nunique()))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--videos", help="only these videos: test, all, or nothing for every result")
    a = ap.parse_args(argv)
    cfg = common.load_config(a.config)
    videos = None
    if a.videos:
        cfg["run"]["videos"] = a.videos
        videos = set(common._questions_and_clips(cfg)[0].video)
    df = summarize(cfg, videos)
    if not len(df):
        sys.exit(f"no results under {cfg['paths']['out_dir']}")
    path = os.path.join(cfg["paths"]["out_dir"], "summary.csv")
    df.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    cell = df.assign(cell=[f"{a:.1f} ({b:.1f}%)" for a, b in zip(df.accuracy, df.budget_pct)])
    print("\naccuracy % (token budget vs Uniform K), rows x models:")
    print(cell.pivot_table(index="condition", columns="model", values="cell", aggfunc="first").fillna("-").to_string())
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
