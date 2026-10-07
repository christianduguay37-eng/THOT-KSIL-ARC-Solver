import json
import numpy as np

with open("training/ecdecbb3.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"][:2]):
    inp = np.array(p["input"])
    out = np.array(p["output"])
    print(f"=== Ex {idx} ===")
    print("IN:")
    for row in inp:
        print("".join(str(v) if v != 0 else "." for v in row))
    print("OUT:")
    for row in out:
        print("".join(str(v) if v != 0 else "." for v in row))
