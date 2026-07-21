import json, glob, collections, sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "results"
if not os.path.isdir(root):
    sys.exit(f"FAIL: '{root}' does not exist — nothing to audit.")

counts = collections.Counter()
bad = []
files = [f for f in glob.glob(f"{root}/**/*.json", recursive=True)
         if not f.endswith("param.json")]
if not files:
    sys.exit(f"FAIL: no detail JSONs under '{root}'.")

for f in files:
    try:
        data = json.load(open(f))
    except Exception:
        counts["unreadable"] += 1; continue
    for s in data.get("step_history", []):
        t = s.get("step_type")
        counts[f"steps:{t}"] += 1
        if t != "sparql":
            continue
        st = s.get("status")
        if st == "timeout":
            counts["timeout"] += 1; bad.append((f, "timeout"))
        elif st and st not in ("ok", "success"):
            q = s.get("sparql", "")
            if "ns:/" in q or "ns:http" in q:
                counts["method_error"] += 1
            else:
                counts["infra_error"] += 1; bad.append((f, st))
        if s.get("result_count") == 1000:
            counts["at_limit_1000"] += 1
print(f"files: {len(files)}")
print(dict(counts))
for f, why in bad[:20]:
    print(f"  {why}: {f}")

if counts.get("steps:sparql", 0) == 0:
    sys.exit("FAIL: zero sparql steps found — agentic runs missing or schema mismatch.")
if counts.get("timeout", 0) or any(k.startswith("status:") for k in counts):
    sys.exit("FAIL: timeouts or errors present — not a clean pass.")
print("PASS: clean.")
