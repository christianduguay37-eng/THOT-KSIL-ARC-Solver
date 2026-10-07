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
        if inp.shape[0] <= 15 and inp.shape[1] <= 15:
            print("IN:\n", inp)
            print("OUT:\n", out)
        else:
            print("IN nonzero:\n", [(r, c, inp[r,c]) for r in range(inp.shape[0]) for c in range(inp.shape[1]) if inp[r,c]!=0])
            print("OUT nonzero:\n", [(r, c, out[r,c]) for r in range(out.shape[0]) for c in range(out.shape[1]) if out[r,c]!=0])
    for i, p in enumerate(d["test"]):
        inp = np.array(p["input"])
        out = np.array(p.get("output", []))
        print(f"Test {i}: in {inp.shape} -> out {out.shape}")
        if inp.shape[0] <= 15 and inp.shape[1] <= 15:
            print("IN:\n", inp)
            if len(out): print("OUT:\n", out)

tids = ["36d67576", "e8593010", "928ad970"]
for tid in tids:
    if os.path.exists(f"training/{tid}.json"):
        examine(tid)
