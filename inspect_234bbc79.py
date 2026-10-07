import json
import numpy as np

with open("training/234bbc79.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"=== Ex {idx} ===")
    print("IN:\n", inp)
    if len(out):
        print("OUT:\n", out)
