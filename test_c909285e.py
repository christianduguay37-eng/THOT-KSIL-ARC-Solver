import json
import numpy as np

def solve_c909285e(inp):
    inp = np.array(inp)
    H, W = inp.shape
    
    for c in np.unique(inp):
        if c == 0:
            continue
        r, col = np.where(inp == c)
        r0, r1 = r.min(), r.max()
        c0, c1 = col.min(), col.max()
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if 4 <= h < H and 4 <= w < W:
            sub = inp[r0:r1+1, c0:c1+1]
            top_frac = np.mean(sub[0, :] == c)
            bot_frac = np.mean(sub[-1, :] == c)
            left_frac = np.mean(sub[:, 0] == c)
            right_frac = np.mean(sub[:, -1] == c)
            if top_frac > 0.8 and bot_frac > 0.8 and left_frac > 0.8 and right_frac > 0.8:
                return sub.copy()
                
    return inp.copy()

if __name__ == "__main__":
    with open("training/c909285e.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_c909285e(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_c909285e(inp)
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
        print("ALL TESTS PASSED for c909285e!")
