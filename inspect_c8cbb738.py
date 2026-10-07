import json
import numpy as np

with open("training/c8cbb738.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"Ex {idx}: in {inp.shape} -> out {out.shape if len(out) else '?'}")
    bg = int(np.argmax(np.bincount(inp.flatten())))
    print(f"  bg: {bg}")
    for c in np.unique(inp):
        if c == bg: continue
        pts = np.argwhere(inp == c)
        print(f"  color {c}: {len(pts)} pts -> {pts.tolist()}")
    if len(out):
        print("  OUT:\n", out)
