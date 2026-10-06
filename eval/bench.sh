#!/usr/bin/env bash
# Throughput by request type against a running server. Writes results/bench_<tag>_<set>.json
# (results/bench_vllm_* are the published numbers).
#   python3 make_tests.py banking77_test.csv tests.json && ./bench.sh default [http://127.0.0.1:8000]
set -euo pipefail
cd "$(dirname "$0")"
TAG=$1; BASE=${2:-http://127.0.0.1:8000}
python3 - <<'PY'
import json
t = json.load(open("tests.json"))
json.dump([r for r in t if r["set"] == "mixed"] * 4, open("short.json", "w"))
json.dump([r for r in t if r["set"] == "banking77"], open("b77.json", "w"))
json.dump([r for r in t if r["set"] == "long"], open("long.json", "w"))
PY
mkdir -p results
for spec in "short 16" "b77 8" "long 4"; do
  set -- $spec
  printf "%-24s" "$TAG $1 c$2:"
  python3 run_tests.py "$1.json" "results/bench_${TAG}_$1.json" "$BASE" "$2"
done
