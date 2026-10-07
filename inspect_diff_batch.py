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

tids = ["8731374e", "97a05b5b", "90c28cc7", "9aec4887", "a8c38be5", "4290ef0e"]
for tid in tids:
    examine(tid)
