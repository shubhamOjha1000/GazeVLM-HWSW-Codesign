# system_perf — latency, memory, FLOPs and energy of every accuracy-table method

The measurements behind `BaseLines_Efficinet Video LLMs/paper_tables/system_performance.tex` (desktop table), for
any of the six VLMs, on exactly the code that produced the accuracy tables: each method is scored by
`accuracy_tables.run.row_jobs` (the same backends, conditions and knobs, e.g. the budget-matched config), **one
condition and one question at a time**, so no cache hides a frame's encoding. Nothing in the models or the
methods is changed: the measurements come from hooks and counters around them.

| file | what it does |
|---|---|
| `measure.py` | runs the measurements, writes `<out-dir>/<model>/{system,flops,batch}.jsonl`, `gate.json`, `run_info.json` |
| `report.py` | the table: mean +- s.d. per (row, condition), speed-ups / savings vs Uniform K, overheads, accuracy joined from the accuracy run; writes `system_table.csv` and `system_table.tex` |
| `probe.py` | the instrumentation: decoder hooks, KV-cache decoding, batch expansion, NVML energy, the in-memory frame cache |
| `colab_system_perf.ipynb` | Colab (A100): tests, measure, report, for one model |
| `colab_system_perf_t4.ipynb` | a small check on a T4: SmolVLM2-2.2B in fp16, one video with all its questions, 1 warm-up |
| `../tests/test_system_perf.py` | CPU tests (sampling, hooks, decoding = recomputation on a tiny Llama, frame cache, report arithmetic) |

## Metrics (per question)

| metric | how |
|---|---|
| **TTFT** (`ttft_s`) | wall clock (`torch.cuda.synchronize()` at the ends) from decoded frames in memory to the first answer token: preprocessing, vision encoder, the method's compression, prompt assembly and the LLM prefill |
| vision side / prefill (`ve_s`, `prefill_s`) | TTFT split at the LLM's first call: a forward pre-hook on the language model's decoder stack marks it |
| **VE memory** (`ve_peak_gb`) | peak allocated GPU memory up to the LLM's first call, minus the memory held before the question (weights): dynamic memory only, as UTOPI |
| **LD memory** (`ld_peak_gb`) | peak during a KV-cache prefill + decoding of the same prompt, minus what was held before it (activations + KV cache) |
| **Overall** (`overall_gb`) | max(VE, LD); the report adds the saving vs Uniform K |
| **TPS** (`tps`) | the answer is one token, so decoding speed is measured separately: prefill with a KV cache from the decoder's captured prefill inputs, then 63 greedy tokens (64 in all, EOS ignored); TPS = 63 / decoding time. Decoding cost depends on the cached prompt length, not on the words, so this stands in for the protocol's open-ended prompt |
| **Frames encoded** (`frames_encoded`) | images the vision encoder sees: every input frame (ours: the gate's frames; the baselines: all Uniform K frames, even those they later drop; the two-view GazeVLM: two per frame) |
| **VE GFLOPs** (`flops.jsonl`) | `torch.utils.flop_counter.FlopCounterMode` over the first sampled question: everything before the LLM's first call (encoder + the method's torch ops), and the prefill. Counted: matmuls, convolutions and attention (the ops with FLOP formulas in torch); not counted: other torch ops (e.g. `cdist`, sorting) and numpy work (gaze heatmaps, fovea layouts) |
| **Method overhead** | ours: the eye + head gate on the question's video stream (`gate.json`: decisions x models as the accuracy run uses them, FLOPs counted, ms timed after a warm-up); baselines (same Uniform K frames as the Uniform K row): their vision side minus Uniform K's on the same question, in FLOPs and ms |
| **Energy / power** (`energy_j`, `power_w`, `decode_energy_j`) | NVML's GPU energy counter over the TTFT window and over the decoding (sampled power where the counter is missing; none without NVML) |
| **Memory vs batch size** (`batch.jsonl`, `--batch-sizes`) | the decoder's peak memory for the first question's prompt repeated B times (prefill + 8 decoded tokens), until the first OOM |

## Protocol

- **Questions**: `--questions 20` sampled round robin over the videos (random within a video, `--seed`), so short and
  long, quiet and busy videos are all represented; the same questions for every condition.
- **Warm-up**: `--warmup 3` untimed runs of the first question per condition.
- **Frames in memory**: every sampled frame is decoded once before timing (`FrameCache`: `cv2.imread` and
  `PIL.Image.open` memoized for paths under `paths.frames_local_dir`), so timings exclude disk reads and JPEG
  decoding. The two-view GazeVLM's gaze crop goes through an uncompressed BMP in `/dev/shm` (part of its timing).
- **Same everything**: batch size 1, the models' SDPA attention, bf16, the conditions of the given config. Use the
  budget-matched config (`accuracy_tables/calibrate.py`) so that the rows are the accuracy table's.
- **Accuracy** comes from the accuracy run on all questions (`--accuracy <summary.csv>` in `report.py`), never from
  the 20 sampled ones.

## Running

```bash
python -m system_perf.measure --config <matched config> --model qwen2.5 --questions 20 \
    --out-dir /content/drive/MyDrive/GazeVLM/system_perf --batch-sizes 1,2,4,8,16,32
python -m system_perf.report --results /content/drive/MyDrive/GazeVLM/system_perf --model qwen2.5 \
    --accuracy /content/drive/MyDrive/GazeVLM/accuracy_tables/qwen25_matched/summary.csv --label "Qwen2.5-VL-3B"
```

Options of `measure.py`: `--rows` (default: every row), `--conditions` (substrings), `--videos`, `--decode-tokens`,
`--no-flops`, `--dtype` (`auto`: bf16 where the GPU has it, fp16 otherwise, e.g. a T4; fp16 is honoured by SmolVLM2,
the other models load bf16). Results append; `report.py` keeps the last record per (row, condition, question).

## Not covered yet

- **Quantized models and the Jetson** (the paper's edge table): the scripts are device-agnostic, but the model
  modules load bf16 weights; 4-bit / 8-bit loading and Jetson power (tegrastats; NVML is missing there) are to add.
- FLOPs of numpy code (gaze heatmap, fovea layout, the methods' small CPU steps) are not counted; their time is in
  the vision side's milliseconds.
