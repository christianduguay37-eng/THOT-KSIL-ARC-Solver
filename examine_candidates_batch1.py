import json, os
import numpy as np

def examine(tid):
    with open(f"training/{tid}.json") as f:
        d = json.load(f)
    print(f"\n==================== TASK {tid} ====================")
    for i, p in enumerate(d["train"]):
        inp = np.array(p["input"])
        out = np.array(p["output"])
        print(f"--- Train {i}: in {inp.shape} -> out {out.shape} ---")
        if max(inp.shape) <= 15 and max(out.shape) <= 15:
            print("IN:\n", inp)
            print("OUT:\n", out)
        else:
            print("IN (first 10x10):\n", inp[:10, :10])
            print("OUT (first 10x10):\n", out[:10, :10])
            print("IN unique:", np.unique(inp), "OUT unique:", np.unique(out))
    for i, p in enumerate(d["test"]):
        inp = np.array(p["input"])
        out = np.array(p.get("output", []))
        print(f"--- Test {i}: in {inp.shape} -> out {out.shape if len(out) else 'unknown'} ---")
        if max(inp.shape) <= 15 and len(out) and max(out.shape) <= 15:
            print("IN:\n", inp)
            print("OUT:\n", out)
        elif len(out):
            print("IN unique:", np.unique(inp), "OUT unique:", np.unique(out))

tids = ["150deff5", "36d67576", "447fd412", "91714a58", "9edfc990", "e8dc4411", "c8cbb738", "9ecd008a", "234bbc79", "b190f7f5"]
for tid in tids:
    examine(tid)
