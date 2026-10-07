import json
import numpy as np

with open("training/a8c38be5.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"=== Ex {idx} ===")
    print(f"in {inp.shape} -> out {out.shape if len(out) else '?'}")
    print("IN colors:", np.unique(inp))
    if len(out):
        print("OUT:\n", out)
