"""Compare two run_tests.py outputs option by option. Usage: compare.py tests.json a.json b.json"""
import json, sys, statistics as st
tests = json.load(open(sys.argv[1])); A = json.load(open(sys.argv[2]))["results"]; B = json.load(open(sys.argv[3]))["results"]
def probs(ans):
    if ans["type"] == "noul": return {"true": ans["noul"], "false": round(1 - ans["noul"], 4)}
    return ans["probabilities"]
def top(p): return max(p, key=p.get)
per = {}
worst = []
for t, a, b in zip(tests, A, B):
    assert a["response"]["usage"] == b["response"]["usage"], (a["response"]["usage"], b["response"]["usage"])
    for qid, qa in a["response"]["answers"].items():
        pa, pb = probs(qa), probs(b["response"]["answers"][qid])
        diff = max(abs(pa[k] - pb[k]) for k in pa); tv = sum(abs(pa[k] - pb[k]) for k in pa) / 2
        s = sorted(pa.values(), reverse=True); margin = s[0] - s[1]
        d = per.setdefault(t["set"], {"n": 0, "agree": 0, "diff": [], "tv": [], "gold_a": 0, "gold_b": 0, "gold_n": 0})
        d["n"] += 1; d["agree"] += top(pa) == top(pb); d["diff"].append(diff); d["tv"].append(tv)
        if top(pa) != top(pb): worst.append((t["set"], qid, top(pa), top(pb), round(margin, 4)))
        if "gold" in t and qid in t["gold"]:
            d["gold_n"] += 1; d["gold_a"] += top(pa) == t["gold"][qid]; d["gold_b"] += top(pb) == t["gold"][qid]
print(f"{'set':10} {'qs':>4} {'top1 agree':>10} {'mean TV':>8} {'max |dp|':>9} {'p99 |dp|':>9}  accuracy A -> B")
for k, d in per.items():
    diffs = sorted(d["diff"])
    acc = f"{d['gold_a']/d['gold_n']:.1%} -> {d['gold_b']/d['gold_n']:.1%}" if d["gold_n"] else ""
    print(f"{k:10} {d['n']:4} {d['agree']/d['n']:10.1%} {st.mean(d['tv']):8.4f} {diffs[-1]:9.4f} {diffs[int(0.99*(len(diffs)-1))]:9.4f}  {acc}")
print("top-1 flips (set, question, A, B, A's top1-top2 margin):", worst or "none")
