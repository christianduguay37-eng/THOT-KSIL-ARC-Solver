import json
import numpy as np

with open("training/39e1d7f9.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"=== Ex {idx} in {inp.shape} ===")
    print("IN colors:", np.unique(inp))
    if len(out):
        diff = (inp != out)
        new_color = np.unique(out[diff])[0]
        pts = np.argwhere(diff)
        print(f"  new color: {new_color}, {len(pts)} pts")
        # Let us see what other colors are in IN
        for c in np.unique(inp):
            if c == 0: continue
            print(f"    color {c} in IN: {(inp == c).sum()} pts")
        # Where are the new color pts placed?
        print("  bbox of new pts:", pts.min(axis=0), "to", pts.max(axis=0))
