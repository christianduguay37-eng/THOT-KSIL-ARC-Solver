import json
import numpy as np

with open("training/9edfc990.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"Ex {idx}: in {inp.shape} -> out {out.shape if len(out) else '?'}")
    print("IN nonzeros:", np.unique(inp))
    if len(out):
        diff = (inp != out)
        print("num diffs:", np.count_nonzero(diff))
        print("diff colors in -> out:")
        for r, c in np.argwhere(diff):
            print(f"  ({r},{c}): {inp[r, c]} -> {out[r, c]}")
