import json
import numpy as np
from scipy.ndimage import label

def solve_b27ca6d3(inp):
    inp = np.array(inp)
    H, W = inp.shape
    out = inp.copy()
    
    lbl2, num2 = label(inp == 2)
    for c in range(1, num2 + 1):
        comp = (lbl2 == c)
        if np.sum(comp) >= 2:
            r, col = np.where(comp)
            r0, r1 = r.min(), r.max()
            c0, c1 = col.min(), col.max()
            for nr in range(r0 - 1, r1 + 2):
                for nc in range(c0 - 1, c1 + 2):
                    if 0 <= nr < H and 0 <= nc < W:
                        if out[nr, nc] == 0:
                            out[nr, nc] = 3
                            
    return out

if __name__ == "__main__":
    with open("training/b27ca6d3.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_b27ca6d3(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_b27ca6d3(inp)
        print(f"Test {i} output shape: {pred.shape}")
        
    if all_ok:
        print("ALL TESTS PASSED for b27ca6d3!")
