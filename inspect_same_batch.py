import json
import numpy as np

def examine(tid):
    with open(f"training/{tid}.json") as f:
        d = json.load(f)
    print(f"\n==================== TASK {tid} ====================")
    for i, p in enumerate(d["train"]):
        inp = np.array(p["input"])
        out = np.array(p["output"])
        print(f"Train {i}: in {inp.shape} -> out {out.shape}")
        if max(inp.shape) <= 12:
            print("IN:\n", inp)
            print("OUT:\n", out)
        else:
            diff = (inp != out)
            print("  diff count:", np.count_nonzero(diff), "IN colors:", np.unique(inp), "OUT colors:", np.unique(out))
    for i, p in enumerate(d["test"]):
        inp = np.array(p["input"])
        out = np.array(p.get("output", []))
        print(f"Test {i}: in {inp.shape} -> out {out.shape if len(out) else '?'}")
        if len(out) and max(inp.shape) <= 12:
            print("IN:\n", inp)
            print("OUT:\n", out)

tids = ["264363fd", "39e1d7f9", "7b6016b9", "83302e8f", "8eb1be9a", "b775ac94", "c444b776", "d22278a0"]
for tid in tids:
    examine(tid)
