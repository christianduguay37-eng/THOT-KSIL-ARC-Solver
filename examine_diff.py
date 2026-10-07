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
        if inp.shape[0] <= 15 and inp.shape[1] <= 15 and out.shape[0] <= 15 and out.shape[1] <= 15:
            print("IN:\n", inp)
            print("OUT:\n", out)
        else:
            print("IN nonzero count:", np.count_nonzero(inp), "OUT shape:", out.shape)
            if out.shape[0] <= 10 and out.shape[1] <= 10:
                print("OUT:\n", out)
    for i, p in enumerate(d["test"]):
        inp = np.array(p["input"])
        out = np.array(p.get("output", []))
        print(f"Test {i}: in {inp.shape} -> out {out.shape}")
        if inp.shape[0] <= 15 and inp.shape[1] <= 15 and len(out) and out.shape[0] <= 15 and out.shape[1] <= 15:
            print("IN:\n", inp)
            print("OUT:\n", out)

tids = ["4290ef0e", "469497ad", "6b9890af", "6ecd11f4", "846bdb03", "8731374e"]
for tid in tids:
    if os.path.exists(f"training/{tid}.json"):
        examine(tid)
