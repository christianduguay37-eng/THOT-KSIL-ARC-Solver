import json
import numpy as np

def solve_5c2c9af4(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    
    pts = list(zip(*np.where(grid != 0)))
    if len(pts) != 3:
        return grid
        
    color = grid[pts[0]]
    pts.sort()
    p1, p2, p3 = pts
    
    r0, c0 = p2
    step = abs(p2[0] - p1[0])
    if step == 0:
        return grid
        
    out = np.zeros((H, W), dtype=int)
    max_k = max(H, W) // step + 3
    
    for k in range(max_k):
        R = k * step
        r_min, r_max = r0 - R, r0 + R
        c_min, c_max = c0 - R, c0 + R
        
        # Top and bottom edges
        for c in range(c_min, c_max + 1):
            if 0 <= c < W:
                if 0 <= r_min < H:
                    out[r_min, c] = color
                if 0 <= r_max < H:
                    out[r_max, c] = color
                    
        # Left and right edges
        for r in range(r_min, r_max + 1):
            if 0 <= r < H:
                if 0 <= c_min < W:
                    out[r, c_min] = color
                if 0 <= c_max < W:
                    out[r, c_max] = color
                    
    return out

if __name__ == "__main__":
    with open("training/5c2c9af4.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_5c2c9af4(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_5c2c9af4(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 5c2c9af4!")
