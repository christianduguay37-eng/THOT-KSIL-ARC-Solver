import json
import numpy as np

def solve_f1cefba8(grid):
    inp = np.array(grid)
    H, W = inp.shape
    out = inp.copy()
    
    vals, counts = np.unique(inp, return_counts=True)
    fg_colors = [c for c in vals if c != 0]
    
    bboxes = {}
    for c in fg_colors:
        rows, cols = np.where(inp == c)
        bboxes[c] = (rows.min(), rows.max(), cols.min(), cols.max())
        
    c_outer = max(fg_colors, key=lambda c: (bboxes[c][1]-bboxes[c][0]) * (bboxes[c][3]-bboxes[c][2]))
    c_inner = [c for c in fg_colors if c != c_outer][0]
    
    R_min, R_max, C_min, C_max = bboxes[c_outer]
    
    inner_rows, inner_cols = np.where(inp == c_inner)
    
    notch_rows = []
    notch_cols = []
    notch_coords = []
    for r, c in zip(inner_rows, inner_cols):
        outer_neighs = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                           if 0 <= r+dr < H and 0 <= c+dc < W and inp[r+dr, c+dc] == c_outer)
        if outer_neighs >= 3:
            notch_coords.append((r, c))
            if (r > 0 and inp[r-1, c] == c_outer and r < H-1 and inp[r+1, c] == c_inner) or \
               (r < H-1 and inp[r+1, c] == c_outer and r > 0 and inp[r-1, c] == c_inner):
                notch_cols.append(c)
            if (c > 0 and inp[r, c-1] == c_outer and c < W-1 and inp[r, c+1] == c_inner) or \
               (c < W-1 and inp[r, c+1] == c_outer and c > 0 and inp[r, c-1] == c_inner):
                notch_rows.append(r)
                
    # Erase the notches in the frame: set to c_outer
    for r, c in notch_coords:
        out[r, c] = c_outer
        
    # Draw laser lines for each notch row
    for r in notch_rows:
        for c in range(W):
            if c < C_min or c > C_max:
                out[r, c] = c_inner
            elif inp[r, c] == c_inner:
                out[r, c] = c_outer
                
    # Draw laser lines for each notch col
    for c in notch_cols:
        for r in range(H):
            if r < R_min or r > R_max:
                out[r, c] = c_inner
            elif inp[r, c] == c_inner:
                out[r, c] = c_outer
                
    return out.tolist()

if __name__ == "__main__":
    with open("training/f1cefba8.json") as f:
        task = json.load(f)
    for idx, ex in enumerate(task["train"]):
        res = solve_f1cefba8(ex["input"])
        assert res == ex["output"], f"Train {idx} failed!"
        print(f"Train {idx} PASS!")
    for idx, ex in enumerate(task["test"]):
        res = solve_f1cefba8(ex["input"])
        print(f"Test {idx} output shape: {len(res)}x{len(res[0])}")
        print("Test 0 SUCCESS!")
