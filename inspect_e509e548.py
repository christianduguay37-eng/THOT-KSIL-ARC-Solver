import json
import numpy as np
from scipy.ndimage import label

with open("training/e509e548.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    lbl, num = label(inp == 3)
    print(f"=== Ex {idx} num comps = {num} ===")
    for c_id in range(1, num + 1):
        mask = (lbl == c_id)
        pts = np.argwhere(mask)
        area = len(pts)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        out_colors = np.unique(out[mask]) if len(out) else []
        print(f"  comp {c_id}: area {area}, bbox ({h}x{w}), out_color = {out_colors}")
