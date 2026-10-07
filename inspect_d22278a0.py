import json
import numpy as np

with open("training/d22278a0.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"=== Ex {idx} in {inp.shape} ===")
    seeds = [(r, c, inp[r, c]) for r in range(inp.shape[0]) for c in range(inp.shape[1]) if inp[r, c] != 0]
    print("  seeds:", seeds)
    if len(out):
        for r, c, val in seeds:
            print(f"  color {val} in OUT: count = {(out == val).sum()}")
