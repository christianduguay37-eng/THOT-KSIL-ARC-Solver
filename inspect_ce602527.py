import json
import numpy as np

with open("training/ce602527.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"=== Ex {idx} in {inp.shape} -> out {out.shape if len(out) else '?'} ===")
    if len(out):
        print("OUT:\n", out)
        # Check if out is a subgrid of inp
        H, W = inp.shape
        h, w = out.shape
        found = []
        for r in range(H - h + 1):
            for c in range(W - w + 1):
                if np.array_equal(inp[r:r+h, c:c+w], out):
                    found.append((r, c))
        print("  found exact match at:", found)
