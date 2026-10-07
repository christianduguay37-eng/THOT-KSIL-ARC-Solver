import json
import numpy as np
from scipy.ndimage import label

def solve_b782dc8a(inp):
    inp = np.array(inp)
    out = inp.copy()
    
    seed_mask = (inp != 0) & (inp != 8)
    if not np.any(seed_mask):
        return out
        
    parity_color = {}
    for r, c in zip(*np.where(seed_mask)):
        p = (r + c) % 2
        parity_color[p] = inp[r, c]
        
    lbl, num = label(inp != 8)
    seed_comps = np.unique(lbl[seed_mask])
    
    target_mask = np.isin(lbl, seed_comps) & (inp == 0)
    for r, c in zip(*np.where(target_mask)):
        p = (r + c) % 2
        if p in parity_color:
            out[r, c] = parity_color[p]
            
    return out

if __name__ == "__main__":
    with open("training/b782dc8a.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_b782dc8a(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_b782dc8a(inp)
        if "output" in ex:
            out = np.array(ex["output"])
            if not np.array_equal(pred, out):
                print(f"FAILED Test {i}")
                all_ok = False
            else:
                print(f"PASS Test {i}")
        else:
            print(f"Test {i} output shape: {pred.shape}")
            
    if all_ok:
        print("ALL TESTS PASSED for b782dc8a!")
