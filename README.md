# Clef on vLLM for the DGX Spark (BF16, unquantized)

This repo runs Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) 27B decision model on a
single NVIDIA DGX Spark (GB10) with vLLM and the original BF16 weights. Nothing is quantized. On
the same machine it's about 1.5× faster than Cloudflare's reference code serving one request at a
time, with 99.4 to 99.7% top-1 agreement and the same BANKING77 accuracy.

> Unofficial. Not affiliated with or endorsed by Cloudflare, NVIDIA, or the plugin's author.

## Why this exists

Clef doesn't generate text. You give it a state and a schema of typed questions (`choice`,
`score`, `noul`), and one prefill pass returns a probability for every allowed option. So the
number that matters is how many records per second you can score.

Cloudflare ships Clef as a Python library on plain transformers. `joint_schema_model.py` provides
`encode_record`, `collate_records` and the model, and there is no server. Two things in it limit
throughput:
- `collate_records` pads every record in a batch to the longest one. Batching a 2k-token record
  with a 60k-token one costs as much as two 60k-token records, and in our measurements padded
  batches also push attention onto a slower masked kernel.
- Requests that arrive separately never get batched together. The `systemone()` helper scores one
  request at a time, so concurrent clients wait in line.

vLLM packs the tokens of every in-flight request into one flat batch with no padding, and
continuous batching keeps the GPU busy as requests come and go. It also runs fused kernels and
CUDA graphs. On a Spark that takes Clef from about 880 tok/s to about 1,370 tok/s. That's close to
the GB10's BF16 compute limit, so unquantized weights can't go much faster.

### Why BF16

Quantized weights approximate the trained ones and usually lose some accuracy. The published Clef
quantizations report scores about 1 to 1.5 points lower on their own evaluations. The community
vLLM integrations that run Clef's joint schema head all ship quantized checkpoints, and as far as
we found, the only way to run the model as Cloudflare trained it was their reference code, one
request at a time. This repo serves the full BF16 release at vLLM speed.

### Credit

The vLLM plugin comes from
[kurcontko/clef-NVFP4](https://huggingface.co/kurcontko/clef-NVFP4). It runs Cloudflare's
unmodified `JointSchemaHead` inside vLLM's pooler. We use the plugin's code but not its quantized
weights, and add:
- a Spark build and launch scripts for the BF16 release;
- a tokenizer thread-safety fix (see [NOTICE](NOTICE));
- the validation and benchmarks below.

## Requirements

- DGX Spark (GB10, 128 GB unified memory). Tested with driver 580.142 and CUDA 13.0.
- Docker with the NVIDIA container toolkit, and a user that can run `docker`.
- About 55 GB of disk for the weights, plus about 21 GB for the image.
- Nothing else using much GPU memory at the same time. The weights take 53 GiB, and the default
  `--gpu-memory-utilization 0.85` gives most of the rest to vLLM. Because GPU and host share one
  memory pool, the host is left with about 10 GB.

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

- To keep the weights somewhere else, run `MODEL_ROOT=/path ./setup.sh` and
  `MODEL_ROOT=/path ./start.sh`.
- `PORT`, `NAME`, `IMAGE` and `GPU_MEMORY_UTILIZATION` are environment variables too.
- Extra arguments to `start.sh` are passed to the server, e.g. `./start.sh --max-num-seqs 128`.
  `docker run --rm clef-vllm:0.28.0 --help` lists them.
- Stop with `docker stop clef-vllm` and restart with `docker start clef-vllm`. No restart policy
  is set.

`setup.sh` builds a `clef-vllm-view/` directory of relative symlinks that puts the joint head under
`clef_head/`, where the plugin looks for it, so the weights aren't copied.

## API

Request and response bodies match the release's `joint_schema_model.systemone()` (the
Jev/SystemOne `POST /v1/systemone` API). The
[Clef model card](https://huggingface.co/Cloudflare/clef) documents the input format.

| Endpoint | |
|---|---|
| `POST /v1/systemone` | one record → `{"model", "answers", "usage"}` |
| `GET /health` | 200 once the engine is up and warmed up |
| `GET /v1/models` | served model name |

Send requests concurrently. vLLM batches whatever is in flight, so a client that waits for each
answer before sending the next gets the single-stream numbers below.

The server accepts text and JSON state only and rejects `images` and `videos`. As in the release's
`encode_record`, a state longer than `--max-model-len` (65,536 here) is truncated to fit.

## Results

All measurements are from one DGX Spark running vLLM 0.28.0 (`vllm/vllm-openai:v0.28.0`, torch
2.13 + cu130). GPU clocks were capped with `nvidia-smi -lgc 300,2200`, a common GB10 stability
setting that held about 2190 MHz under load. Uncapped clocks should be somewhat faster.

The reference is the release's own transformers code (`joint_schema_model.py`) in BF16, with
transformers 5.18, torch 2.14.1 and flash-linear-attention 0.5.2. It ran behind a minimal HTTP
wrapper, one request at a time.

### Test set

`eval/make_tests.py` builds the test set deterministically with seed 1234:

| Set | Records | Questions | Tokens / record |
|---|---:|---:|---:|
| `banking77` | 150 [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) test messages, one 77-way `choice` with gold labels | 150 | ~1,830 |
| `mixed` | 50 messages wrapped in JSON, with a `noul`, `score`, 4-way `choice`, `noul` each | 200 | ~450 |
| `long` | transcripts of 7.5k, 14.9k, 29.6k and 55.2k tokens with a planted fact, 2 questions each | 8 | — |

### Agreement with the reference

We compare probabilities option by option. TV is the mean total-variation distance per question.

| Comparison | Top-1 agreement | Mean TV (banking77 / mixed / long) | BANKING77 accuracy |
|---|---:|---|---:|
| reference vs vLLM, 1 at a time | 357/358 (99.7%) | 0.0031 / 0.0052 / 0.0022 | 94.7% → 94.7% |
| reference vs vLLM, 8 concurrent | 356/358 (99.4%) | 0.0036 / 0.0051 / 0.0023 | 94.7% → 94.7% |
| vLLM 1 at a time vs vLLM 8 concurrent | 357/358 (99.7%) | 0.0024 / 0.0033 / 0.0015 | — |

The median per-option difference is 0.0009. The answers that flipped were near-ties, where the
reference's top two options were 0.003 and 0.025 apart. The largest single difference was 0.16, on
one 4-way `choice` where the top option went from 0.61 to 0.77.

vLLM differs from itself across batch compositions (the last row) by about as much as it differs
from the reference. We attribute the drift to BF16 rounding in different kernels and batch shapes.

For comparison, Cloudflare reports 94.2% BANKING77 macro-F1 for Clef. The reference got 94.7%
accuracy on our 150-message sample.

### Throughput

| Workload | Reference (1 at a time) | vLLM |
|---|---:|---:|
| `mixed`, ~450-token records | 1.94 req/s, 878 tok/s | **3.03 req/s**, 1,374 tok/s (16 concurrent) |
| `banking77`, ~1.8k tokens | 0.52 req/s, 945 tok/s | **0.77 req/s**, 1,411 tok/s (8 concurrent) |
| `long`, 7.5k–55k tokens | 745 tok/s | **1,252 tok/s** (4 concurrent) |
| whole test set | 0.44 req/s, 879 tok/s | 0.64 req/s, 1,279 tok/s (1 at a time); 0.69 req/s, 1,363 tok/s (8 concurrent) |

### Headroom

Clef is prefill-only and dense, so BF16 matrix-multiply speed sets its throughput. At Clef's
shapes, cuBLAS BF16 GEMMs on the GB10 run at about 86 to 95 TFLOPS (see `eval/kernels/`). About
1,400 tok/s works out to roughly 75 TFLOPS of useful work, or about 80% of that peak.

Getting much faster on one Spark would take quantization; FP8 GEMMs measure about 183 TFLOPS here.
This repo keeps the release's BF16 weights on purpose. For quantized options, see the
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
is CC-BY-4.0, so the scripts download it instead of redistributing it.

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
