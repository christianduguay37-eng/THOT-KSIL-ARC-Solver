import json
import numpy as np

with open("training/e73095fd.json") as f:
    d = json.load(f)

for idx, p in enumerate(d["train"] + d["test"]):
    inp = np.array(p["input"])
    out = np.array(p.get("output", []))
    print(f"=== Ex {idx} in {inp.shape} ===")
    if len(out):
        diff = (inp != out)
        print("num diffs:", np.count_nonzero(diff))
        # diff transitions:
        for val_in in np.unique(inp[diff]):
            for val_out in np.unique(out[diff & (inp == val_in)]):
                cnt = np.count_nonzero(diff & (inp == val_in) & (out == val_out))
                print(f"  {val_in} -> {val_out}: {cnt}")
