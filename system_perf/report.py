"""The system-performance table from system_perf/measure.py's results (mean +- s.d. over the sampled questions).

    python -m system_perf.report --results /content/drive/MyDrive/GazeVLM/system_perf --model qwen2.5 \
        --accuracy /content/drive/MyDrive/GazeVLM/accuracy_tables/qwen25_matched/summary.csv

Per (row, condition): budget (visual tokens vs the Uniform K row, same questions), frames encoded, vision-side
GFLOPs, VE / LD / overall peak dynamic memory (and the saving vs Uniform K), method overhead, TTFT and TPS (and the
speed-ups vs Uniform K), energy and power, accuracy (joined by condition name from the accuracy run's summary.csv).
Method overhead: ours = the eye + head gate on the question's video stream (gate.json: FLOPs and ms; ours' foveation
layout is a few hundred cells of numpy per frame, counted in its vision-side time); the baselines, which all encode
the same Uniform K frames, = their vision side minus Uniform K's on the same question (FLOPs and ms). Writes
<results>/<model>/system_table.csv and system_table.tex (rows in the layout of system_performance.tex).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

METRICS = ("ttft_s", "ve_s", "prefill_s", "tps", "decode_s", "ve_peak_gb", "ld_peak_gb", "overall_gb", "energy_j",
           "power_w", "decode_energy_j")
BASE_ROWS = ("uniform", "blind", "ours")


def _jsonl(path):
    return pd.DataFrame([json.loads(l) for l in open(path) if l.strip()]) if os.path.exists(path) else pd.DataFrame()


def load(results, model):
    d = os.path.join(results, model)
    gate_path = os.path.join(d, "gate.json")
    gate = json.load(open(gate_path)) if os.path.exists(gate_path) else {}
    sysdf = _jsonl(os.path.join(d, "system.jsonl"))
    if len(sysdf) and "error" in sysdf:
        sysdf = sysdf[sysdf["error"].isna()]
    sysdf = sysdf.drop_duplicates(["row", "condition", "qid"], keep="last") if len(sysdf) else sysdf
    return sysdf, _jsonl(os.path.join(d, "flops.jsonl")), gate, _jsonl(os.path.join(d, "batch.jsonl"))


def table(sysdf, flops, gate, acc=None):
    """-> DataFrame, one row per (row, condition)."""
    if not len(sysdf):
        return pd.DataFrame()
    for m in METRICS:
        if m not in sysdf:
            sysdf[m] = np.nan
        sysdf[m] = pd.to_numeric(sysdf[m], errors="coerce")
    uni = sysdf[sysdf.row == "uniform"].set_index("qid")
    fl = flops.drop_duplicates(["row", "condition"], keep="last").set_index("condition") if len(flops) else pd.DataFrame()
    uni_flops = fl[fl.row == "uniform"].ve_gflops.mean() if len(fl) and (fl.row == "uniform").any() else np.nan
    u = {m: uni[m].mean() for m in METRICS} if len(uni) else {}
    out = []
    for (row, cond), g in sysdf.groupby(["row", "condition"], sort=False):
        g = g.set_index("qid")
        q = g.index.intersection(uni.index)
        r = dict(row=row, condition=cond, n=len(g),
                 budget_pct=100 * g.loc[q, "n_image_tokens"].mean() / uni.loc[q, "n_image_tokens"].mean() if len(q) else np.nan,
                 frames_encoded=g.frames_encoded.mean(), image_tokens=g.n_image_tokens.mean())
        for m in METRICS:
            r[m], r[m + "_sd"] = g[m].mean(), g[m].std(ddof=1) if len(g) > 1 else 0.0
        r["ve_gflops"] = fl.loc[cond, "ve_gflops"] if len(fl) and cond in fl.index else np.nan
        r["prefill_gflops"] = fl.loc[cond, "prefill_gflops"] if len(fl) and cond in fl.index else np.nan
        if row == "ours":
            vids = g.video.tolist()
            r["overhead_mflops"] = np.mean([gate[v]["flops"] / 1e6 for v in vids if v in gate]) if gate else np.nan
            r["overhead_ms"] = np.mean([gate[v]["ms"] for v in vids if v in gate]) if gate else np.nan
        elif row in BASE_ROWS:
            r["overhead_mflops"] = r["overhead_ms"] = np.nan
        else:
            r["overhead_mflops"] = 1000 * (r["ve_gflops"] - uni_flops)
            r["overhead_ms"] = 1000 * (g.loc[q, "ve_s"].mean() - uni.loc[q, "ve_s"].mean()) if len(q) else np.nan
        if u:
            r["ttft_speedup"] = u["ttft_s"] / r["ttft_s"] if r["ttft_s"] else np.nan
            r["tps_speedup"] = r["tps"] / u["tps"] if u["tps"] else np.nan
            r["overall_saving_pct"] = 100 * (1 - r["overall_gb"] / u["overall_gb"]) if u["overall_gb"] else np.nan
        if acc is not None and cond in acc.index:
            r["accuracy"] = float(acc.loc[cond, "accuracy"])
        out.append(r)
    return pd.DataFrame(out)


def _f(x, fmt="{:.2f}"):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


def latex_rows(df, model_label):
    """Rows in system_performance.tex's desktop-table layout."""
    lines = []
    for i, r in df.iterrows():
        name = {"uniform": "stock (uniform $K$)", "ours": r"\textbf{Ours}"}.get(r.row, r.row)
        lines.append(" & ".join([
            model_label if not lines else "", f"{name}", _f(r.budget_pct, "{:.1f}\\%"), _f(r.frames_encoded, "{:.0f}"),
            _f(r.ve_gflops, "{:.0f}"), _f(r.ve_peak_gb), _f(r.ld_peak_gb),
            f"{_f(r.overall_gb)} ({_f(r.get('overall_saving_pct'), '{:.0f}')}\\%)",
            f"{_f(r.overhead_mflops, '{:.1f}')} / {_f(r.overhead_ms, '{:.0f}')}",
            f"{_f(r.ttft_s)} ({_f(r.get('ttft_speedup'), '{:.2f}')}$\\times$)",
            f"{_f(r.tps, '{:.1f}')} ({_f(r.get('tps_speedup'), '{:.2f}')}$\\times$)",
            _f(r.get("accuracy"), "{:.1f}")]) + r" \\")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="measure.py's --out-dir")
    ap.add_argument("--model", default="qwen2.5")
    ap.add_argument("--accuracy", help="the accuracy run's summary.csv (accuracy per condition)")
    ap.add_argument("--label", help="model name in the LaTeX rows (default: --model)")
    a = ap.parse_args(argv)
    sysdf, flops, gate, batch = load(a.results, a.model)
    acc = None
    if a.accuracy and os.path.exists(a.accuracy):
        acc = pd.read_csv(a.accuracy)
        acc = acc[acc.model == a.model].drop_duplicates("condition").set_index("condition")
    df = table(sysdf, flops, gate, acc)
    if not len(df):
        raise SystemExit(f"no results in {os.path.join(a.results, a.model)}")
    d = os.path.join(a.results, a.model)
    df.to_csv(os.path.join(d, "system_table.csv"), index=False)
    with open(os.path.join(d, "system_table.tex"), "w") as f:
        f.write(latex_rows(df, a.label or a.model) + "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["row", "condition", "n", "budget_pct", "frames_encoded", "ve_gflops", "ve_peak_gb", "ld_peak_gb", "overall_gb",
            "overall_saving_pct", "overhead_mflops", "overhead_ms", "ttft_s", "ttft_sd", "ttft_speedup", "tps",
            "tps_speedup", "energy_j", "power_w", "accuracy"]
    df = df.rename(columns={"ttft_s_sd": "ttft_sd"})
    print(df[[c for c in cols if c in df]].round(2).to_string(index=False))
    if gate:
        print(f"\ngate (ours' decision overhead), per video: " + ", ".join(
            f"{v}: {g['decisions']} decisions, {g['flops'] / 1e6:.2f} MFLOPs, {g['ms']:.1f} ms" for v, g in gate.items()))
    if len(batch):
        print("\ndecoder peak memory (GB) vs batch size:")
        for _, r in batch.iterrows():
            print(f"   {r.row:<13} {r.condition[:60]:<60} {r.peak_gb}")
    print(f"\nwritten {d}/system_table.csv and system_table.tex")


if __name__ == "__main__":
    main()
