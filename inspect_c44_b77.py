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
        # print nonzeros
        diff = (inp != out)
        print("  diff count:", np.count_nonzero(diff))
        print("  diff pts:", np.argwhere(diff)[:5].tolist(), "...")
        print("  IN nonzeros:", np.unique(inp), "OUT nonzeros:", np.unique(out))

examine("c444b776")
examine("b775ac94")
