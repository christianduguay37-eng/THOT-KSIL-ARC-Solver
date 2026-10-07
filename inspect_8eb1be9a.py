import json
import numpy as np

with open("training/8eb1be9a.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"=== Ex {idx} in {inp.shape} ===")
    print("IN nonzeros:")
    for r in range(inp.shape[0]):
        if np.any(inp[r] != 0):
            print(f"  row {r}: {inp[r].tolist()}")
    if len(out):
        print("OUT nonzeros:")
        for r in range(out.shape[0]):
            print(f"  row {r}: {out[r].tolist()}")
