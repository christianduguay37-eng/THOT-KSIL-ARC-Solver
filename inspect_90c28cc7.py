import json
import numpy as np

with open("training/90c28cc7.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"=== Ex {idx} ===")
    print("IN nonzeros:", np.unique(inp))
    if len(out):
        print("OUT:\n", out)
        # Check where each cell of out comes from in inp
        H, W = out.shape
        for r in range(H):
            for c in range(W):
                color = out[r, c]
                pts = np.argwhere(inp == color)
                print(f"  out[{r},{c}] = {color}: {len(pts)} pts in inp, e.g. {pts[:3].tolist()}")
