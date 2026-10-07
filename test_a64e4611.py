import json
import numpy as np
from scipy.ndimage import binary_dilation

def solve_a64e4611(inp):
    inp = np.array(inp)
    H, W = inp.shape
    cand = ~binary_dilation(inp != 0, structure=np.ones((3, 3)))
    
    best_strip = None
    best_area = -1
    for c0 in range(W):
        for c1 in range(c0 + 2, W + 1):
            r0 = H
            while r0 > 0 and np.all(cand[r0 - 1:H, c0:c1]):
                r0 -= 1
            area = (H - r0) * (c1 - c0)
            if area > best_area and (H - r0) >= 10:
                best_area = area
                best_strip = (r0, H, c0, c1)
                
    if best_strip is None:
        return inp.copy()
        
    rv0, rv1, cv0, cv1 = best_strip
    out = inp.copy()
    out[rv0:rv1, cv0:cv1] = 3
    
    for r in range(H):
        if np.all(cand[r, 0:cv0]):
            out[r, 0:cv1] = 3
        if np.all(cand[r, cv1:W]):
            out[r, cv0:W] = 3
            
    full_horiz_rows = [r for r in range(H) if np.all(out[r, :] == 3)]
    if full_horiz_rows:
        r_top = min(full_horiz_rows)
        r_bot = max(full_horiz_rows)
        for c in range(W):
            if np.all(cand[0:r_top+1, c]):
                out[0:r_top+1, c] = 3
        for c in range(W):
            if np.all(cand[r_bot:H, c]):
                out[r_bot:H, c] = 3
                
    return out

if __name__ == "__main__":
    with open("training/a64e4611.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_a64e4611(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_a64e4611(inp)
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
        print("ALL TESTS PASSED for a64e4611!")
