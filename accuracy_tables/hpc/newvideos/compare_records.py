"""Do two scoring runs give the same answers? Used to check that loading frames on threads changes nothing.

    python -u -m accuracy_tables.hpc.newvideos.compare_records --new <folder> --ref <folder> --rows budget,nogate

Compares each record in <new> with the same (condition, question) in <ref>: predicted answer, frames, visual tokens and the
largest probability difference. Ends with SAME ANSWERS, or a failed job. Prints no accuracy.

With --report, for runs on different GPU types: only different frames or tokens fail the job, and it says how many answers
differ and how many records <ref> doesn't have, for the user to judge.
"""
import argparse
import json
import os
import sys


def load_records(path):
    """{(condition, qid): record}"""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["condition"], r["qid"]): r for r in rows}


def compare_records(new, ref):
    """-> dict(records, missing, pred_differs, tokens_differ, max_prob_diff)."""
    out = dict(records=len(new), missing=0, pred_differs=0, tokens_differ=0, max_prob_diff=0.0)
    for key, a in new.items():
        b = ref.get(key)
        if b is None:
            out["missing"] += 1
            continue
        out["pred_differs"] += a.get("pred") != b.get("pred")
        out["tokens_differ"] += (a.get("n_image_tokens"), a.get("n_frames")) != (b.get("n_image_tokens"), b.get("n_frames"))
        if a.get("p") and b.get("p"):
            out["max_prob_diff"] = max(out["max_prob_diff"], max(abs(x - y) for x, y in zip(a["p"], b["p"])))
    return out


def problems(r, report=False):
    """How many differences in compare_records' result fail the job."""
    return r["tokens_differ"] + (r["records"] == 0) + (0 if report else r["missing"] + r["pred_differs"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--rows", required=True, help="comma list of result files, without .jsonl")
    ap.add_argument("--report", action="store_true", help="only different frames or tokens fail; report different answers")
    a = ap.parse_args()
    bad = compared = changed = 0
    for row in a.rows.split(","):
        r = compare_records(load_records(os.path.join(a.new, f"{row}.jsonl")), load_records(os.path.join(a.ref, f"{row}.jsonl")))
        print(f"{row}: {r}")
        bad += problems(r, a.report)
        compared += r["records"] - r["missing"]
        changed += r["pred_differs"]
    if a.report:
        if bad:
            sys.exit("TOKENS DIFFER: stop and tell the user")
        print(f"TOKENS MATCH: answers differ on {changed} of {compared} records compared. Report this to the user.")
    elif bad:
        sys.exit("ANSWERS DIFFER: stop and tell the user")
    else:
        print("SAME ANSWERS")


if __name__ == "__main__":
    main()
