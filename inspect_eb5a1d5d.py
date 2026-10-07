import json
import numpy as np

with open("training/eb5a1d5d.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"=== Ex {idx} ===")
    print("IN nonzeros:", np.unique(inp))
    # counts of each color in IN
    for c in np.unique(inp):
        if c == 0: continue
        print(f"  color {c}: count = {(inp == c).sum()}")
    if len(out):
        # order of colors from outside to inside:
        order = []
        S = out.shape[0]
        for k in range(S // 2 + 1):
            order.append(out[k, k])
        print("  OUT colors from outside to inside:", order)
