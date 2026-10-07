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
        else:
            print("IN nonzero count:", np.count_nonzero(inp), "colors:", np.unique(inp))
            print("OUT nonzero count:", np.count_nonzero(out), "colors:", np.unique(out))
    for i, p in enumerate(d["test"]):
        inp = np.array(p["input"])
        out = np.array(p.get("output", []))
        print(f"Test {i}: in {inp.shape} -> out {out.shape if len(out) else 'None'}")
        if max(inp.shape) <= 12 and len(out) and max(out.shape) <= 12:
            print("IN:\n", inp)
            print("OUT:\n", out)
        elif len(out):
            print("IN nonzero count:", np.count_nonzero(inp), "colors:", np.unique(inp))
            print("OUT nonzero count:", np.count_nonzero(out), "colors:", np.unique(out))

tids = ["447fd412", "91714a58", "40853293", "484b58aa", "5c2c9af4", "6455b5f5", "6aa20dc0", "6ecd11f4"]
for tid in tids:
    examine(tid)
