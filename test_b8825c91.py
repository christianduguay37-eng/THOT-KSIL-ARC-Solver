import json
import numpy as np

def solve_b8825c91(inp):
    inp = np.array(inp)
    H, W = inp.shape
    out = inp.copy()
    
    def get_syms(r, c):
        return [
            (r, c),
            (r, W - 1 - c),
            (H - 1 - r, c),
            (H - 1 - r, W - 1 - c),
            (c, r),
            (c, W - 1 - r),
            (W - 1 - c, r),
            (W - 1 - c, H - 1 - r),
        ]
        
    for r in range(H):
        for c in range(W):
            if out[r, c] == 4:
                vals = [inp[nr, nc] for nr, nc in get_syms(r, c) if inp[nr, nc] != 4]
                if vals:
                    out[r, c] = vals[0]
                    
    return out

if __name__ == "__main__":
    with open("training/b8825c91.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_b8825c91(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_b8825c91(inp)
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
        print("ALL TESTS PASSED for b8825c91!")
