import json
import numpy as np

def solve_40853293(grid):
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    
    colors = [c for c in np.unique(grid) if c != 0]
    
    horiz_segments = []
    vert_segments = []
    
    for c in colors:
        pts = list(zip(*np.where(grid == c)))
        if len(pts) == 2:
            (r0, c0), (r1, c1) = pts
            if r0 == r1:
                horiz_segments.append((r0, min(c0, c1), max(c0, c1), c))
            elif c0 == c1:
                vert_segments.append((c0, min(r0, r1), max(r0, r1), c))
                
    # Draw horizontal segments first
    for r, c_start, c_end, col in horiz_segments:
        for c in range(c_start, c_end + 1):
            out[r, c] = col
            
    # Draw vertical segments on top
    for c, r_start, r_end, col in vert_segments:
        for r in range(r_start, r_end + 1):
            out[r, c] = col
            
    return out

if __name__ == "__main__":
    with open("training/40853293.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_40853293(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_40853293(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 40853293!")
