"""The budget rule against the no-gate control, budget by budget, on the same questions.

    python -m accuracy_tables.compare_budget --model smolvlm --out-dir <results folder>

Prints and writes <out_dir>/<model>/budget_compare.csv: accuracy difference (rule - no gate) in points, its 95% interval
from resampling whole videos, and both rows' frames per question and measured token budgets. `outcome` says "rule ahead"
or "no gate ahead" only if the interval excludes 0. This is only the comparison: it doesn't judge the rule on its own,
and on the 12 AEA videos it only checks the code, because those videos are too short for the limit to thin them much.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accuracy_tables import summarize  # noqa: E402

_NAME = re.compile(r"^(budget rule|no gate) ([0-9.]+)% ")


def compare(res, B=2000):
    """res: result rows with row, condition, qid, video, correct, n_frames, n_image_tokens -> one row per budget."""
    from vlmeval import egoev
    conds = {}
    for c in res.condition.unique():
        m = _NAME.match(c)
        if m:
            conds.setdefault(float(m.group(2)), {})[m.group(1)] = c
    bud = summarize.budgets(res)
    frames = res.groupby("condition").n_frames.mean()
    out = []
    for pct in sorted(conds):
        rule, nogate = conds[pct].get("budget rule"), conds[pct].get("no gate")
        if not rule or not nogate:
            continue
        t = egoev.paired_test(res, rule, nogate, B)
        outcome = "rule ahead" if t["ci_low"] > 0 else "no gate ahead" if t["ci_high"] < 0 else "no clear difference"
        out.append(dict(budget=pct, n=t["n"], rule_minus_nogate=t["diff"], ci_low=t["ci_low"], ci_high=t["ci_high"],
                        rule_only_right=t["a_only_right"], nogate_only_right=t["b_only_right"],
                        rule_frames=frames[rule], nogate_frames=frames[nogate],
                        rule_budget_pct=bud.get(rule, np.nan), nogate_budget_pct=bud.get(nogate, np.nan),
                        outcome=outcome))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True, help="the results folder (holds <model>/budget.jsonl and nogate.jsonl)")
    a = ap.parse_args(argv)
    res = summarize.load_model_results(a.out_dir, a.model)
    table = compare(res) if len(res) else pd.DataFrame()
    if not len(table):
        sys.exit(f"{a.out_dir}/{a.model}: no budget and nogate results to compare")
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))
    path = os.path.join(a.out_dir, a.model, "budget_compare.csv")
    table.to_csv(path, index=False)
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
