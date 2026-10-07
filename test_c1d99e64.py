import json
import numpy as np

def solve_c1d99e64(inp):
    inp = np.array(inp)
    H, W = inp.shape
    out = inp.copy()
    
    empty_rows = [r for r in range(H) if np.all(inp[r, :] == 0)]
    empty_cols = [c for c in range(W) if np.all(inp[:, c] == 0)]
    
    for r in empty_rows:
        out[r, :] = 2
    for c in empty_cols:
        out[:, c] = 2
        
    return out

if __name__ == "__main__":
    with open("training/c1d99e64.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_c1d99e64(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_c1d99e64(inp)
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
        print("ALL TESTS PASSED for c1d99e64!")
