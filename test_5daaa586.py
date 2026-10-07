import json
import numpy as np

def solve_5daaa586(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    
    v_lines = []
    for c in range(W):
        col = grid[:, c]
        for color in range(1, 10):
            if (col == color).sum() >= H - 2:
                v_lines.append((c, color))
                break
                
    h_lines = []
    for r in range(H):
        row = grid[r, :]
        for color in range(1, 10):
            if (row == color).sum() >= W - 2:
                h_lines.append((r, color))
                break
                
    v_lines.sort()
    h_lines.sort()
    
    if len(v_lines) < 2 or len(h_lines) < 2:
        return grid
        
    c_min, col_left_color = v_lines[0]
    c_max, col_right_color = v_lines[1]
    r_min, row_top_color = h_lines[0]
    r_max, row_bottom_color = h_lines[1]
    
    sub = grid[r_min:r_max+1, c_min:c_max+1].copy()
    sub_H, sub_W = sub.shape
    
    interior = sub[1:-1, 1:-1]
    interior_colors = [c for c in np.unique(interior) if c != 0]
    if not interior_colors:
        return sub
    dot_color = interior_colors[0]
    
    if row_top_color == dot_color:
        for c in range(1, sub_W - 1):
            col_dots = [r for r in range(1, sub_H - 1) if sub[r, c] == dot_color]
            if col_dots:
                max_r = max(col_dots)
                for r in range(0, max_r + 1):
                    sub[r, c] = dot_color
    elif row_bottom_color == dot_color:
        for c in range(1, sub_W - 1):
            col_dots = [r for r in range(1, sub_H - 1) if sub[r, c] == dot_color]
            if col_dots:
                min_r = min(col_dots)
                for r in range(min_r, sub_H):
                    sub[r, c] = dot_color
    elif col_left_color == dot_color:
        for r in range(1, sub_H - 1):
            row_dots = [c for c in range(1, sub_W - 1) if sub[r, c] == dot_color]
            if row_dots:
                max_c = max(row_dots)
                for c in range(0, max_c + 1):
                    sub[r, c] = dot_color
    elif col_right_color == dot_color:
        for r in range(1, sub_H - 1):
            row_dots = [c for c in range(1, sub_W - 1) if sub[r, c] == dot_color]
            if row_dots:
                min_c = min(row_dots)
                for c in range(min_c, sub_W):
                    sub[r, c] = dot_color
                    
    return sub

if __name__ == "__main__":
    with open("training/5daaa586.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_5daaa586(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_5daaa586(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 5daaa586!")
