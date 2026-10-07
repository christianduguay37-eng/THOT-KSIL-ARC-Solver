import json
import numpy as np

def solve_aba27056(inp):
    inp = np.array(inp)
    H, W = inp.shape
    out = inp.copy()
    
    r_nz, c_nz = np.where(inp != 0)
    if len(r_nz) == 0:
        return out
    r0, r1 = r_nz.min(), r_nz.max()
    c0, c1 = c_nz.min(), c_nz.max()
    
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if out[r, c] == 0:
                out[r, c] = 4
                
    top_open = [c for c in range(c0, c1 + 1) if inp[r0, c] == 0]
    bot_open = [c for c in range(c0, c1 + 1) if inp[r1, c] == 0]
    left_open = [r for r in range(r0, r1 + 1) if inp[r, c0] == 0]
    right_open = [r for r in range(r0, r1 + 1) if inp[r, c1] == 0]
    
    if top_open:
        c_min, c_max = min(top_open), max(top_open)
        for k in range(1, r0 + 1):
            r = r0 - k
            for c in range(c_min, c_max + 1):
                if 0 <= c < W: out[r, c] = 4
            if 0 <= c_min - k < W: out[r, c_min - k] = 4
            if 0 <= c_max + k < W: out[r, c_max + k] = 4
    elif bot_open:
        c_min, c_max = min(bot_open), max(bot_open)
        for k in range(1, H - r1):
            r = r1 + k
            for c in range(c_min, c_max + 1):
                if 0 <= c < W: out[r, c] = 4
            if 0 <= c_min - k < W: out[r, c_min - k] = 4
            if 0 <= c_max + k < W: out[r, c_max + k] = 4
    elif left_open:
        r_min, r_max = min(left_open), max(left_open)
        for k in range(1, c0 + 1):
            c = c0 - k
            for r in range(r_min, r_max + 1):
                if 0 <= r < H: out[r, c] = 4
            if 0 <= r_min - k < H: out[r_min - k, c] = 4
            if 0 <= r_max + k < H: out[r_max + k, c] = 4
    elif right_open:
        r_min, r_max = min(right_open), max(right_open)
        for k in range(1, W - c1):
            c = c1 + k
            for r in range(r_min, r_max + 1):
                if 0 <= r < H: out[r, c] = 4
            if 0 <= r_min - k < H: out[r_min - k, c] = 4
            if 0 <= r_max + k < H: out[r_max + k, c] = 4
            
    return out

if __name__ == "__main__":
    with open("training/aba27056.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_aba27056(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_aba27056(inp)
        print(f"Test {i} output shape: {pred.shape}")
        
    if all_ok:
        print("ALL TESTS PASSED for aba27056!")
