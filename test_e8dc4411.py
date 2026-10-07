import json
import numpy as np

def solve_e8dc4411(grid):
    inp = np.array(grid)
    H, W = inp.shape
    out = inp.copy()
    
    vals, counts = np.unique(inp, return_counts=True)
    bg = vals[np.argmax(counts)]
    fg_col = [c for c in vals if c != bg and c != 0][0]
    
    zero_coords = list(zip(*np.where(inp == 0)))
    fg_in = list(zip(*np.where(inp == fg_col)))[0]
    
    # Find direction: which (+-1, +-1) connects a zero pixel to fg_in?
    dir_r, dir_c = None, None
    for dr in [-1, 1]:
        for dc in [-1, 1]:
            if (fg_in[0] - dr, fg_in[1] - dc) in zero_coords:
                dir_r, dir_c = dr, dc
                break
                
    # Opposite extremal point
    candidates = sorted(zero_coords, key=lambda pt: pt[0] * dir_r + pt[1] * dir_c)
    step_r, step_c = None, None
    for pt in candidates:
        s_r = fg_in[0] - pt[0]
        s_c = fg_in[1] - pt[1]
        if abs(s_r) == abs(s_c) and s_r * dir_r > 0:
            step_r, step_c = s_r, s_c
            break
            
    # Shift zero_coords repeatedly by (step_r, step_c)
    k = 1
    while True:
        any_in_bounds = False
        for r, c in zero_coords:
            nr = r + k * step_r
            nc = c + k * step_c
            if 0 <= nr < H and 0 <= nc < W:
                any_in_bounds = True
                out[nr, nc] = fg_col
        if not any_in_bounds:
            break
        k += 1
        
    return out.tolist()

if __name__ == "__main__":
    with open("training/e8dc4411.json") as f:
        task = json.load(f)
    for idx, ex in enumerate(task["train"]):
        res = solve_e8dc4411(ex["input"])
        assert res == ex["output"], f"Train {idx} failed!"
        print(f"Train {idx} PASS!")
    for idx, ex in enumerate(task["test"]):
        res = solve_e8dc4411(ex["input"])
        print(f"Test {idx} output shape: {len(res)}x{len(res[0])}")
        print("Test 0 SUCCESS!")
