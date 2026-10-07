import json
import numpy as np

with open("training/b190f7f5.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    H, W = inp.shape
    print(f"Ex {idx}: shape {inp.shape} -> out {out.shape if len(out) else '?'}")
    # Half width is H
    # Left half: inp[:, :H], Right half: inp[:, H:]
    left = inp[:, :H]
    right = inp[:, H:]
    print(f"  left has 8: {8 in left}, right has 8: {8 in right}")
