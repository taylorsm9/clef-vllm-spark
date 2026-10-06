# Clef on vLLM for the DGX Spark (BF16, unquantized)

A ready-to-run server for Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) 27B decision
model on a single NVIDIA DGX Spark (GB10), using vLLM and the **original BF16 weights**. Nothing
is quantized. It's about **1.5× faster** than serving Cloudflare's reference code one request at a
time on the same machine, and the answers are effectively unchanged: 99.4–99.7% top-1 agreement
and identical BANKING77 accuracy.

> Unofficial. Not affiliated with or endorsed by Cloudflare, NVIDIA, or the plugin's author.

## Why this exists

Clef doesn't generate text. It reads a state and a schema of typed questions (`choice`, `score`,
`noul`) and returns a probability for every allowed option, in one prefill pass. That makes it a
throughput workload. What matters is how many records per second you can score.

**Cloudflare's release is a library, not a server.** `joint_schema_model.py` gives you
`encode_record`, `collate_records` and the model, on plain transformers:
- `collate_records` batches by **padding every record to the longest one**. A 2k-token record
  batched with a 60k-token one costs as much as two 60k-token records, and on our measurements
  padded batches also drop attention onto a slower masked kernel.
- Nothing batches requests that arrive separately. The `systemone()` helper scores one request
  at a time, so concurrent clients queue behind each other.

**vLLM fixes both.**
- It packs every in-flight request's tokens into one flat batch, with no padding.
- It keeps the GPU full with continuous batching.
- It runs fused kernels and CUDA graphs.

On a Spark that turns about 880 tok/s into about 1,370 tok/s. The model sits close to the GB10's
BF16 compute limit, so it's not far from the ceiling for unquantized weights.

**Why BF16.** Quantized weights are approximations of the trained ones, and they usually cost
some accuracy. The published Clef quantizations report this themselves: about 1–1.5 points lower
on their evaluations. The community vLLM integrations that run Clef's joint schema head all
ship with such quantized checkpoints. As far as we found, the only way to run the model as Cloudflare
trained it was their reference code, one request at a time. This repo serves the full BF16
release at vLLM speed.

This repo builds on the small vLLM plugin published with
[kurcontko/clef-NVFP4](https://huggingface.co/kurcontko/clef-NVFP4). It runs Cloudflare's
unmodified `JointSchemaHead` inside vLLM's pooler. We use only the plugin's code, not its
quantized weights, and add:
- a Spark build and launch scripts for the BF16 release;
- a tokenizer thread-safety fix (see [NOTICE](NOTICE));
- the validation and benchmarks below.

## Requirements

- DGX Spark (GB10, 128 GB unified memory). Tested with driver 580.142 and CUDA 13.0.
- Docker with the NVIDIA container toolkit, and a user that can run `docker`.
- About 55 GB of disk for the weights, plus about 21 GB for the image.
- Nothing else that uses much GPU memory running at the same time. Weights take 53 GiB, and the
  default `--gpu-memory-utilization 0.85` hands most of the rest to vLLM. GPU and host share one
  memory pool, so this leaves the host about 10 GB.

## Quick start

```bash
./setup.sh                    # build image clef-vllm:0.28.0, download Cloudflare/clef @ 2f3de3dd
./start.sh                    # detached container "clef-vllm" on :8000; 3–4 min until healthy
curl -s localhost:8000/health

curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "clef",
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle the message?",
                   "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is a service down?"}}}'
```

- Use `MODEL_ROOT=/path ./setup.sh` and `MODEL_ROOT=/path ./start.sh` to keep the weights elsewhere.
- `PORT`, `NAME`, `IMAGE` and `GPU_MEMORY_UTILIZATION` are also environment variables.
- Extra arguments to `start.sh` go to the server, e.g. `./start.sh --max-num-seqs 128`. Run
  `docker run --rm clef-vllm:0.28.0 --help` to list them.
- Stop with `docker stop clef-vllm` and restart with `docker start clef-vllm`. No restart policy
  is set.

`setup.sh` builds a `clef-vllm-view/` directory of relative symlinks that puts the joint head under
`clef_head/`, where the plugin expects it, so the weights aren't copied.

## API

The request and response bodies are those of the release's `joint_schema_model.systemone()`
(the Jev/SystemOne `POST /v1/systemone` API). See the
[Clef model card](https://huggingface.co/Cloudflare/clef) for the input format.

| Endpoint | |
|---|---|
| `POST /v1/systemone` | one record → `{"model", "answers", "usage"}` |
| `GET /health` | 200 once the engine is up and warmed up |
| `GET /v1/models` | served model name |

**Send requests concurrently.** vLLM batches whatever is in flight. A client that waits for each
answer before sending the next gets the single-stream numbers below.

**Notes:**
- Text and JSON state only. The server rejects `images` and `videos`.
- As in the release's `encode_record`, a state longer than `--max-model-len` (65,536 here) is
  **truncated** to fit.

## Results

All measurements below are on one DGX Spark:
- GPU clocks were capped with `nvidia-smi -lgc 300,2200` (about 2190 MHz under load), a common
  GB10 stability setting. Uncapped clocks should be somewhat faster.
- vLLM 0.28.0 (`vllm/vllm-openai:v0.28.0`, torch 2.13 + cu130).

The **reference** is the release's own transformers code (`joint_schema_model.py`):
- transformers 5.18, torch 2.14.1, flash-linear-attention 0.5.2, BF16.
- It ran behind a minimal HTTP wrapper, one request at a time.

### Test set

`eval/make_tests.py` builds it deterministically, with seed 1234:

| Set | Records | Questions | Tokens / record |
|---|---:|---:|---:|
| `banking77` | 150 [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) test messages, one 77-way `choice` with gold labels | 150 | ~1,830 |
| `mixed` | 50 messages wrapped in JSON, with a `noul`, `score`, 4-way `choice`, `noul` each | 200 | ~450 |
| `long` | transcripts of 7.5k, 14.9k, 29.6k and 55.2k tokens with a planted fact, 2 questions each | 8 | — |

### Agreement with the reference

Probabilities are compared option by option. "TV" is the mean total-variation distance per
question.

| Comparison | Top-1 agreement | Mean TV (banking77 / mixed / long) | BANKING77 accuracy |
|---|---:|---|---:|
| reference vs vLLM, 1 at a time | 357/358 (99.7%) | 0.0031 / 0.0052 / 0.0022 | 94.7% → 94.7% |
| reference vs vLLM, 8 concurrent | 356/358 (99.4%) | 0.0036 / 0.0051 / 0.0023 | 94.7% → 94.7% |
| vLLM 1 at a time vs vLLM 8 concurrent | 357/358 (99.7%) | 0.0024 / 0.0033 / 0.0015 | — |

- The median per-option difference is 0.0009.
- The flipped answers were near-ties, with reference top-1/top-2 margins of 0.003 and 0.025.
- The largest single difference was 0.16, on one 4-way `choice` (0.61 → 0.77 for the same top
  option).
- vLLM's own variation between batch compositions (last row) is the same size as its difference
  from the reference. We read the drift as BF16 numerics from different kernels and batch shapes,
  not as an error. Cloudflare reports 94.2% BANKING77 macro-F1 for Clef; the reference here got
  94.7% accuracy on our 150-message sample.

### Throughput

| Workload | Reference (1 at a time) | vLLM |
|---|---:|---:|
| `mixed`, ~450-token records | 1.94 req/s, 878 tok/s | **3.03 req/s**, 1,374 tok/s (16 concurrent) |
| `banking77`, ~1.8k tokens | 0.52 req/s, 945 tok/s | **0.77 req/s**, 1,411 tok/s (8 concurrent) |
| `long`, 7.5k–55k tokens | 745 tok/s | **1,252 tok/s** (4 concurrent) |
| whole test set | 0.44 req/s, 879 tok/s | 0.64 req/s, 1,279 tok/s (1 at a time); 0.69 req/s, 1,363 tok/s (8 concurrent) |

**Headroom.**
- Clef is prefill-only and dense, so throughput is bound by BF16 matrix-multiply speed.
- On the GB10, cuBLAS BF16 GEMMs at Clef's shapes run at about 86–95 TFLOPS (`eval/kernels/`).
- About 1,400 tok/s is roughly 75 TFLOPS of useful work, about 80% of that peak.

Going substantially faster on one Spark means quantizing. FP8 GEMMs measure about 183 TFLOPS
here. This repo deliberately keeps the release's BF16 weights; for quantized options, see the
[community quantizations](https://huggingface.co/models?other=base_model:quantized:Cloudflare/clef).

## Reproducing the evaluation

```bash
cd eval
curl -sLO https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv
python3 make_tests.py test.csv tests.json                                    # 204 records
python3 run_tests.py tests.json results/mine_c1.json http://127.0.0.1:8000 1  # or 8 for concurrency
python3 compare.py tests.json results/reference_transformers_c1.json results/mine_c1.json
./bench.sh mine http://127.0.0.1:8000                                         # throughput by workload
```

`eval/results/` holds the raw responses from every run above, including the reference. BANKING77
is CC-BY-4.0 and is downloaded, not redistributed.

## Layout

```
Dockerfile           vllm/vllm-openai:v0.28.0 + the plugin
setup.sh, start.sh   build/download, run
vllm_plugin/         clef-vllm plugin (kurcontko, Apache-2.0) with small changes, see NOTICE
eval/                test set builder, runner, comparison, throughput bench, kernel microbenchmarks
eval/results/        raw results behind the tables above
```

## License

Apache-2.0 ([LICENSE](LICENSE)). Clef and the clef-vllm plugin are also Apache-2.0; see
[NOTICE](NOTICE).
