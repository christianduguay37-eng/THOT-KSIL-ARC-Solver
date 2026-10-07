import json
import numpy as np

with open("training/469497ad.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"Ex {idx}: in {inp.shape} -> out {out.shape if len(out) else '?'}")
    print("IN:\n", inp)
    if len(out):
        print("OUT:\n", out)
