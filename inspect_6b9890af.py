import json
import numpy as np

with open("training/6b9890af.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"=== Ex {idx} in {inp.shape} -> out {out.shape if len(out) else '?'} ===")
    if len(out):
        print("OUT border color:", out[0, 0])
        # search for out inside inp
        H, W = inp.shape
        h, w = out.shape
        for r in range(H - h + 1):
            for c in range(W - w + 1):
                if np.array_equal(inp[r:r+h, c:c+w], out):
                    print(f"  EXACT MATCH of OUT in IN at ({r}, {c})")
        # search for inner of out (without border of 2s)
        inner = out[1:-1, 1:-1]
        ih, iw = inner.shape
        for r in range(H - ih + 1):
            for c in range(W - iw + 1):
                if np.array_equal(inp[r:r+ih, c:c+iw], inner):
                    print(f"  EXACT MATCH of inner in IN at ({r}, {c})")
