"""Send each test record to a /v1/systemone endpoint; save responses + timing. Usage: run_tests.py tests.json out.json base_url [concurrency]"""
import json, sys, time, urllib.request, concurrent.futures as cf
tests, out, base = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
conc = int(sys.argv[4]) if len(sys.argv) > 4 else 1
def call(rec):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(rec["request"]).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r: body = json.loads(r.read())
    return {"response": body, "seconds": time.perf_counter() - t}
t0 = time.perf_counter()
with cf.ThreadPoolExecutor(conc) as ex: results = list(ex.map(call, tests))
wall = time.perf_counter() - t0
toks = sum(r["response"]["usage"]["input_tokens"] for r in results)
json.dump({"wall": wall, "concurrency": conc, "results": results}, open(out, "w"))
print(f"{len(results)} requests, {toks} tokens, wall {wall:.1f}s, {toks/wall:.0f} tok/s, {len(results)/wall:.2f} req/s")
