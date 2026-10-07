import json
import numpy as np

def solve_dc0a314f(grid):
    inp = np.array(grid)
    H, W = inp.shape
    
    # Find 5x5 block of occluder color (unique 5x5 uniform square)
    found = False
    for r in range(H - 5 + 1):
        for c in range(W - 5 + 1):
            sub = inp[r:r+5, c:c+5]
            if len(np.unique(sub)) == 1:
                r0, c0 = r, c
                occl_color = sub[0, 0]
                found = True
                break
        if found:
            break
            
    out = np.zeros((5, 5), dtype=int)
    for dr in range(5):
        for dc in range(5):
            r = r0 + dr
            c = c0 + dc
            candidates = [
                (r, W - 1 - c),
                (H - 1 - r, c),
                (H - 1 - r, W - 1 - c),
                (c, r),
                (c, H - 1 - r),
                (W - 1 - c, r),
                (W - 1 - c, H - 1 - r)
            ]
            for cr, cc in candidates:
                if 0 <= cr < H and 0 <= cc < W and inp[cr, cc] != occl_color:
                    out[dr, dc] = inp[cr, cc]
                    break
                    
    return out.tolist()

if __name__ == "__main__":
    with open("training/dc0a314f.json") as f:
        task = json.load(f)
    for idx, ex in enumerate(task["train"]):
        res = solve_dc0a314f(ex["input"])
        assert res == ex["output"], f"Train {idx} failed!"
        print(f"Train {idx} PASS!")
    for idx, ex in enumerate(task["test"]):
        res = solve_dc0a314f(ex["input"])
        print(f"Test {idx} output shape: {len(res)}x{len(res[0])}")
        print("Test 0 SUCCESS!")
