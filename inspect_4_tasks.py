import json
import numpy as np

def examine(tid):
    with open(f"training/{tid}.json") as f:
        d = json.load(f)
    print(f"\n==================== TASK {tid} ====================")
    for i, p in enumerate(d["train"]):
        inp = np.array(p["input"])
        out = np.array(p["output"])
        print(f"Train {i}: in {inp.shape} -> out {out.shape}")
        diff = (inp != out)
        print("  diff count:", np.count_nonzero(diff))
        # transitions
        transitions = {}
        for r, c in np.argwhere(diff):
            k = (int(inp[r, c]), int(out[r, c]))
            transitions[k] = transitions.get(k, 0) + 1
        print("  transitions:", transitions)

tids = ["b775ac94", "c444b776", "d22278a0", "39e1d7f9"]
for tid in tids:
    examine(tid)
