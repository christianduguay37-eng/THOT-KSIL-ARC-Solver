import json
import numpy as np

def examine(tid):
    with open(f"training/{tid}.json") as f:
        d = json.load(f)
    print(f"=== {tid} ===")
    for i, p in enumerate(d["train"] + d["test"]):
        inp = np.array(p["input"])
        out = np.array(p.get("output", []))
        print(f"Example {i}: in {inp.shape} -> out {out.shape if len(out) else 'None'}")
        if inp.shape[0] <= 15 and inp.shape[1] <= 15:
            print("IN:\n", inp)
            if len(out):
                print("OUT:\n", out)

examine("36d67576")
