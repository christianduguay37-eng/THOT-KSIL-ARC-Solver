import json
import numpy as np

def solve_a8d7556c(inp):
    inp = np.array(inp)
    H, W = inp.shape
    out = inp.copy()
    
    zero_rects = []
    for r0 in range(H - 1):
        for c0 in range(W - 1):
            for r1 in range(r0 + 1, H):
                for c1 in range(c0 + 1, W):
                    if np.all(inp[r0:r1+1, c0:c1+1] == 0):
                        zero_rects.append((r0, r1, c0, c1, (r1 - r0 + 1) * (c1 - c0 + 1)))
                        
    zero_rects.sort(key=lambda x: x[4], reverse=True)
    
    filled = np.zeros((H, W), dtype=bool)
    for r0, r1, c0, c1, area in zero_rects:
        if not np.any(filled[r0:r1+1, c0:c1+1]):
            filled[r0:r1+1, c0:c1+1] = True
            out[r0:r1+1, c0:c1+1] = 2
            
    return out

if __name__ == "__main__":
    with open("training/a8d7556c.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_a8d7556c(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_a8d7556c(inp)
        print(f"Test {i} output shape: {pred.shape}")
        
    if all_ok:
        print("ALL TESTS PASSED for a8d7556c!")
