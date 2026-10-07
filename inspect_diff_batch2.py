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
        if max(inp.shape) <= 12 and max(out.shape) <= 12:
            print("IN:\n", inp)
            print("OUT:\n", out)
        elif max(out.shape) <= 10:
            print("OUT:\n", out)
    for i, p in enumerate(d["test"]):
        inp = np.array(p["input"])
        out = np.array(p.get("output", []))
        print(f"Test {i}: in {inp.shape} -> out {out.shape if len(out) else '?'}")
        if len(out) and max(out.shape) <= 10:
            print("OUT:\n", out)

tids = ["234bbc79", "6b9890af", "c909285e", "eb5a1d5d", "dc0a314f", "ff805c23"]
for tid in tids:
    examine(tid)
