import json
import numpy as np

def examine(tid):
    with open(f"training/{tid}.json") as f:
        d = json.load(f)
    print(f"\n==================== TASK {tid} ====================")
    for i, p in enumerate(d["train"][:2]):
        inp = np.array(p["input"])
        out = np.array(p["output"])
        print(f"Train {i}: in {inp.shape} -> out {out.shape}")
        # print crop of 10x10
        print("IN crop (8x8):\n", inp[:8, :8])
        print("OUT crop (8x8):\n", out[:8, :8])
        print("IN colors:", np.unique(inp), "OUT colors:", np.unique(out))
        # transitions
        for c_in in np.unique(inp):
            for c_out in np.unique(out[inp == c_in]):
                print(f"  {c_in} -> {c_out}: {(out[inp == c_in] == c_out).sum()}")

examine("7b6016b9")
examine("83302e8f")
