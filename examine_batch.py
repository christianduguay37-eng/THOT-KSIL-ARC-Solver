import os, json
import numpy as np

def examine(tid):
    with open(f"training/{tid}.json") as f:
        d = json.load(f)
    print(f"=== TASK {tid} ===")
    for i, p in enumerate(d["train"]):
        inp = np.array(p["input"])
        out = np.array(p["output"])
        print(f"Train {i}: in {inp.shape} -> out {out.shape}")
        if inp.shape[0] <= 15 and inp.shape[1] <= 15:
            print("IN:\n", inp)
            print("OUT:\n", out)
        else:
            print("IN colors:", np.unique(inp), "OUT colors:", np.unique(out))
    for i, p in enumerate(d["test"]):
        inp = np.array(p["input"])
        out = np.array(p.get("output", []))
        print(f"Test {i}: in {inp.shape} -> out {out.shape}")
        if inp.shape[0] <= 15 and inp.shape[1] <= 15:
            print("IN:\n", inp)
            if len(out): print("OUT:\n", out)

# Let's inspect a set of candidate tasks
tids = [
    "150deff5", "36d67576", "3e980e27", "447fd412", "543a7ed5",
    "d22278a0", "e8593010", "928ad970", "e21d9049", "d687bc17",
    "264363fd", "272f95fa", "2dd70a9a"
]

for tid in tids:
    if os.path.exists(f"training/{tid}.json"):
        examine(tid)
