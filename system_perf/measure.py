"""System-performance measurements for one VLM: every method of accuracy_tables/, one condition and one question at
a time, through exactly the code that scores the accuracy tables (accuracy_tables.run.row_jobs).

    python -m system_perf.measure --config <accuracy config, e.g. the budget-matched one> --model qwen2.5 \
        --questions 20 --out-dir /content/drive/MyDrive/GazeVLM/system_perf

Per (row, condition, question), in <out-dir>/<model>/system.jsonl (see README.md for the protocol):
  ttft_s        decoded frames in memory -> the first answer token: preprocessing, vision encoder, the method's
                compression, prompt assembly and the LLM prefill (disk reads and JPEG decoding excluded: FrameCache)
  ve_s          the vision side alone (everything before the LLM's first call); prefill_s the prefill itself
  ve_peak_gb    peak dynamic GPU memory on the vision side, ld_peak_gb in prefill + decoding (KV cache included),
                both above the memory held before the question (weights); overall_gb = their max
  tps           decoding speed: 64 greedy tokens after the same prompt from a fresh KV-cache prefill (EOS ignored)
  energy_j, power_w over the TTFT window, decode_energy_j over the decoding (NVML; None without it)
  frames_encoded, n_frames (in the prompt), n_image_tokens, n_tokens
Per (row, condition), once, in flops.jsonl: ve_gflops (vision side incl. the method's ops) and prefill_gflops,
counted by torch.utils.flop_counter on the first sampled question. In gate.json: the eye + head gate's cost per
video (decisions, FLOPs, ms), ours' only decision overhead. With --batch-sizes, batch.jsonl: the decoder's peak memory
at each batch size (prefill + decoding of the same prompt repeated), OOM recorded as such.
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accuracy_tables import common  # noqa: E402
from accuracy_tables import run as acc_run  # noqa: E402
from accuracy_tables.models import ADAPTERS  # noqa: E402
from system_perf import probe as pr  # noqa: E402

DEFAULT_ROWS = "uniform,blind,ours,gazevlm,gazevlm2,avgpool,streamingtom,dycoke,fastvid,prunevid,dtd"
GB = 1e9


# ---------------------------------------------------------------------------- questions
def sample_questions(qs, n, seed=0):
    """n question ids spread over the videos (round robin over videos in name order, random within a video):
    latency depends on the frames, so every video (short / long, quiet / busy) must be represented."""
    rng = np.random.default_rng(seed)
    pools = {v: list(rng.permutation(sorted(g.qid.astype(int)))) for v, g in qs.groupby("video")}
    out = []
    while len(out) < n and any(pools.values()):
        for v in sorted(pools):
            if pools[v] and len(out) < n:
                out.append(int(pools[v].pop(0)))
    return out


def frames_encoded(row, item):
    """Images the vision encoder sees for a question: every input frame (the two-view GazeVLM: two per frame)."""
    if row == "blind":
        return 0
    return len(item["frames"]) * (2 if row == "gazevlm2" else 1)


# ---------------------------------------------------------------------------- one question
def _score_once(job, cond, item, path):
    if os.path.exists(path):
        os.remove(path)
    job["score"]([item], {cond: job["conds"][cond]}, path)
    lines = [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
    return lines[-1] if lines else {}


def measure_once(model, decoder, job, cond, item, path, decode_tokens=64):
    """One timed question -> the system.jsonl record."""
    probe = pr.Probe(decoder)
    pr.sync()
    torch.cuda.empty_cache()
    base = pr.mem_allocated()
    pr.reset_peak()
    with probe, pr.Energy() as e:
        t0 = time.perf_counter()
        rec = _score_once(job, cond, item, path)
    if probe.t_start is None:
        raise RuntimeError(f"{job['row']} / {cond}: the language model was never called")
    out = dict(row=job["row"], condition=cond, qid=int(item["qid"]), video=item["video"],
               frames_encoded=frames_encoded(job["row"], item), n_frames=rec.get("n_frames"),
               n_image_tokens=rec.get("n_image_tokens"), n_tokens=rec.get("n_tokens"), error=rec.get("error"),
               ve_s=probe.t_start - t0, prefill_s=probe.t_end - probe.t_start, ttft_s=probe.t_end - t0,
               ve_peak_gb=max(0, probe.ve_peak - base) / GB, energy_j=e.joules, power_w=e.watts)
    kwargs = probe.kwargs
    del probe
    torch.cuda.empty_cache()
    base = pr.mem_allocated()
    pr.reset_peak()
    try:
        with pr.Energy() as e:
            d = pr.decode_benchmark(model, decoder, kwargs, decode_tokens)
        out.update(tps=d["tps"], decode_s=d["decode_s"], ld_peak_gb=max(0, pr.mem_peak() - base) / GB,
                   decode_energy_j=e.joules, decode_power_w=e.watts)
    except torch.cuda.OutOfMemoryError:
        out.update(tps=None, decode_s=None, ld_peak_gb=None, decode_error="oom")
    out["overall_gb"] = max(v for v in (out["ve_peak_gb"], out.get("ld_peak_gb")) if v is not None)
    return out, kwargs


def count_flops(decoder, job, cond, item, path):
    """(vision-side FLOPs incl. the method's ops, prefill FLOPs) of one question."""
    probe = pr.Probe(decoder)
    with pr.flop_counter() as fc:
        probe.flop_counter = fc
        with probe:
            _score_once(job, cond, item, path)
        total = fc.get_total_flops()
    before = probe.flops_before if probe.flops_before is not None else total
    return before, total - before


@torch.no_grad()
def batch_memory(model, decoder, kwargs, sizes, decode_tokens=8):
    """{B: decoder peak memory (GB) for the prompt repeated B times, or 'oom'}."""
    out = {}
    for B in sizes:
        torch.cuda.empty_cache()
        base = pr.mem_allocated()
        pr.reset_peak()
        try:
            pr.decode_benchmark(model, decoder, pr.expand_batch(kwargs, B), decode_tokens)
            out[B] = (pr.mem_peak() - base) / GB
        except torch.cuda.OutOfMemoryError:
            out[B] = "oom"
            torch.cuda.empty_cache()
            break
    return out


# ---------------------------------------------------------------------------- ours' decision overhead
def gate_cost(cfg, videos, device=None):
    """The eye + head gate on each video's whole sensor stream, as the accuracy run uses it (test videos: the
    5-fold ensemble; other videos: the fold model that held them out) -> {video: dict(decisions, models, flops,
    ms)}: FLOPs counted by torch.utils.flop_counter, ms after one warm-up."""
    from eyehead.infer import load_ckpt
    from eyehead.oof import _rows, _score
    p = cfg["paths"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cks = [load_ckpt(os.path.join(p["gate_ckpt_dir"], f"fold{i}.pt"), device) for i in range(1, 6)]
    test = _rows(os.path.join(p["folds_dir"], "test.csv"))
    folds = [_rows(os.path.join(p["folds_dir"], f"train_val_fold{i}.csv")) for i in range(1, 6)]
    out = {}
    for v in videos:
        if (test.sequence == v).any():
            df, models = test[test.sequence == v].reset_index(drop=True), cks
        else:
            i = next(i for i, f in enumerate(folds) if ((f.sequence == v) & (f["split"] == "val")).any())
            f = folds[i]
            df, models = f[(f.sequence == v) & (f["split"] == "val")].reset_index(drop=True), [cks[i]]
        run = lambda: [_score(ck, df, device) for ck in models]
        run()                                                          # warm-up
        pr.sync()
        t0 = time.perf_counter()
        run()
        pr.sync()
        ms = 1000 * (time.perf_counter() - t0)
        with pr.flop_counter() as fc:
            run()
        out[v] = dict(decisions=int(len(df)), models=len(models), flops=int(fc.get_total_flops()), ms=ms)
    return out


# ---------------------------------------------------------------------------- the whole measurement
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="an accuracy_tables config (the budget-matched one: same conditions)")
    ap.add_argument("--model", default="qwen2.5", help=f"one of {', '.join(ADAPTERS)}")
    ap.add_argument("--rows", default=DEFAULT_ROWS, help=f"comma list of rows (default: {DEFAULT_ROWS})")
    ap.add_argument("--conditions", help="comma list of substrings: only conditions whose name contains one")
    ap.add_argument("--videos", help="override run.videos: test, all, or a comma list of video names")
    ap.add_argument("--questions", type=int, default=20, help="questions sampled across the videos (default 20)")
    ap.add_argument("--warmup", type=int, default=3, help="untimed runs of the first question per condition (default 3)")
    ap.add_argument("--decode-tokens", type=int, default=64, help="greedy tokens for TPS (default 64)")
    ap.add_argument("--batch-sizes", default="", help="e.g. 1,2,4,8,16,32: decoder memory vs batch size (first question)")
    ap.add_argument("--no-flops", action="store_true", help="skip the FLOP counting pass")
    ap.add_argument("--out-dir", required=True, help="results: <out-dir>/<model>/{system,flops,batch}.jsonl, gate.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16"),
                    help="weights' dtype: auto = bf16 on GPUs with bf16 (A100), fp16 otherwise (e.g. T4); fp16 is "
                         "honoured by SmolVLM2 (VLMEVAL_DTYPE), the other models load bf16")
    a = ap.parse_args(argv)
    dtype = a.dtype
    if dtype == "auto":
        dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "float16"
    os.environ["VLMEVAL_DTYPE"] = dtype
    if dtype == "float16" and a.model != "smolvlm":
        print(f"note: {a.model} loads bf16 whatever --dtype says (only SmolVLM2 honours float16)")

    cfg = common.load_config(a.config)
    if a.videos:
        cfg["run"]["videos"] = a.videos if a.videos in ("test", "all") else [v.strip() for v in a.videos.split(",")]
    rows = [r.strip() for r in a.rows.split(",") if r.strip()]
    out_dir = os.path.join(os.path.abspath(os.path.expanduser(a.out_dir)), a.model)
    os.makedirs(out_dir, exist_ok=True)

    data = common.load_data(cfg)
    qids = sample_questions(data["qs"], a.questions, a.seed)
    print(common.describe(data))
    print(f"{len(qids)} questions sampled across {data['qs'][data['qs'].qid.isin(qids)].video.nunique()} videos: {qids}")

    gate = gate_cost(cfg, sorted(data["qs"].video.unique()))
    with open(os.path.join(out_dir, "gate.json"), "w") as f:
        json.dump(gate, f, indent=1)
    for v, g in gate.items():
        print(f"   gate {v}: {g['decisions']} decisions x {g['models']} model(s), {g['flops'] / 1e6:.2f} MFLOPs, {g['ms']:.1f} ms")

    ad = ADAPTERS[a.model](cfg["models"].get(a.model), cfg).load()
    try:
        acc_run.check_parity(ad, data, a.model)
        decoder = pr.find_decoder(ad.model)
        jobs = acc_run.row_jobs(ad, cfg, data, rows)
        frames_root = cfg["paths"]["frames_local_dir"]
        sys_path, flops_path, batch_path = (os.path.join(out_dir, f) for f in ("system.jsonl", "flops.jsonl", "batch.jsonl"))
        tmp = os.path.join(tempfile.mkdtemp(prefix="sysperf_"), "rec.jsonl")
        with pr.FrameCache(frames_root) as cache, open(sys_path, "a") as fs, open(flops_path, "a") as ff, \
                open(batch_path, "a") as fb:
            paths = sorted({fr["path"] for job in jobs for it in job["items"] if int(it["qid"]) in set(qids)
                            for fr in it["frames"]})
            print(f"decoding {len(paths)} frames into memory (timings exclude disk reads and JPEG decoding) ...")
            cache.preload(paths)
            for job in jobs:
                by_q = {int(it["qid"]): it for it in job["items"]}
                conds = [c for c in job["conds"] if not a.conditions or any(s in c for s in a.conditions.split(","))]
                for cond in conds:
                    print(f"\n--- {job['row']}: {cond}")
                    first = by_q[qids[0]]
                    for _ in range(a.warmup):
                        _score_once(job, cond, first, tmp)
                    kwargs0 = None
                    for q in qids:
                        rec, kw = measure_once(ad.model, decoder, job, cond, by_q[q], tmp, a.decode_tokens)
                        kwargs0 = kwargs0 if kwargs0 is not None else kw
                        rec.update(model=a.model, dtype=dtype,
                                   device=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")
                        fs.write(json.dumps(rec) + "\n")
                        fs.flush()
                        print(f"   q{q}: ttft {rec['ttft_s']:.2f} s (vision {rec['ve_s']:.2f}), tps {rec.get('tps') or 0:.1f}, "
                              f"VE {rec['ve_peak_gb']:.2f} GB, LD {rec.get('ld_peak_gb') or 0:.2f} GB, "
                              f"{rec['frames_encoded']} frames encoded, {rec['n_image_tokens']} tokens")
                    if not a.no_flops:
                        ve, pf = count_flops(decoder, job, cond, first, tmp)
                        ff.write(json.dumps(dict(model=a.model, row=job["row"], condition=cond, qid=qids[0],
                                                 ve_gflops=ve / 1e9, prefill_gflops=pf / 1e9)) + "\n")
                        ff.flush()
                        print(f"   FLOPs (q{qids[0]}): vision side {ve / 1e9:.1f} G, prefill {pf / 1e9:.1f} G")
                    if a.batch_sizes and kwargs0 is not None:
                        sizes = [int(b) for b in a.batch_sizes.split(",") if b.strip()]
                        mem = batch_memory(ad.model, decoder, kwargs0, sizes)
                        fb.write(json.dumps(dict(model=a.model, row=job["row"], condition=cond, qid=qids[0],
                                                 peak_gb={str(k): v for k, v in mem.items()})) + "\n")
                        fb.flush()
                        print(f"   decoder memory vs batch size: {mem}")
                    del kwargs0
                    torch.cuda.empty_cache()
    finally:
        info = dict(model=a.model, model_id=getattr(ad, "model_id", None), config=a.config, questions=qids,
                    rows=rows, warmup=a.warmup, decode_tokens=a.decode_tokens, dtype=dtype,
                    device=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                    torch=torch.__version__)
        try:
            import transformers
            info["transformers"] = transformers.__version__
        except Exception:
            pass
        with open(os.path.join(out_dir, "run_info.json"), "w") as f:
            json.dump(info, f, indent=1)
        ad.free()
    print(f"\nwritten {out_dir}")


if __name__ == "__main__":
    main()
