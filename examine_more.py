import os, json
import numpy as np

def examine(tid):
    with open(f"training/{tid}.json") as f:
        d = json.load(f)
    print(f"\n==================== TASK {tid} ====================")
    for i, p in enumerate(d["train"]):
        inp = np.array(p["input"])
        out = np.array(p["output"])
        print(f"Train {i}: in {inp.shape} -> out {out.shape}")
        print("IN:\n", inp)
        print("OUT:\n", out)
    for i, p in enumerate(d["test"]):
        inp = np.array(p["input"])
        out = np.array(p.get("output", []))
        print(f"Test {i}: in {inp.shape} -> out {out.shape}")
        print("IN:\n", inp)
        if len(out): print("OUT:\n", out)

tids = ["150deff5", "3e980e27", "447fd412", "543a7ed5", "d22278a0"]
for tid in tids:
    if os.path.exists(f"training/{tid}.json"):
        examine(tid)
