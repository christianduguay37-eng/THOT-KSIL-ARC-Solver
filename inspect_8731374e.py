import json
import numpy as np

with open("training/8731374e.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"=== Ex {idx} ===")
    print(f"in {inp.shape} -> out {out.shape if len(out) else '?'}")
    print("IN colors:", np.unique(inp))
    if len(out):
        print("OUT colors:", np.unique(out))
        print("OUT:\n", out)
