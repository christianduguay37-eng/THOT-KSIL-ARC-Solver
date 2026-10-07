import json
import numpy as np
from scipy.ndimage import label

def solve_6a1e5592(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    if 5 not in grid or 2 not in grid:
        return None
    
    labeled5, num5 = label(grid == 5, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    if num5 == 0 or num5 > 6:
        return None
    shapes = []
    for c_id in range(1, num5 + 1):
        pts = list(zip(*np.where(labeled5 == c_id)))
        min_r = min(r for r, c in pts)
        min_c = min(c for r, c in pts)
        norm_pts = set((r - min_r, c - min_c) for r, c in pts)
        shapes.append(norm_pts)
        
    ceiling_2 = set(zip(*np.where(grid == 2)))
    
    candidate_placements = []
    for s_idx, shape in enumerate(shapes):
        valid_pos = []
        for r_top in range(0, 6):
            for c_left in range(0, W):
                placed = set((r_top + r, c_left + c) for r, c in shape)
                if any(r >= H or c >= W for r, c in placed):
                    continue
                if placed & ceiling_2:
                    continue
                up_placed = set((r - 1, c) for r, c in placed)
                if up_placed & ceiling_2:
                    valid_pos.append((r_top, c_left, placed))
        candidate_placements.append(valid_pos)
        
    best_combo = None
    best_score = float('inf')
    
    def search(idx, current_cells, chosen):
        nonlocal best_combo, best_score
        if idx == len(shapes):
            score = sum(r_top for r_top, c_left, _ in chosen)
            if score < best_score:
                best_score = score
                best_combo = list(chosen)
            return
            
        for r_top, c_left, placed in candidate_placements[idx]:
            if not (placed & current_cells):
                cur_score = sum(r for r, _, _ in chosen) + r_top
                if cur_score < best_score:
                    search(idx + 1, current_cells | placed, chosen + [(r_top, c_left, placed)])
                    
    search(0, set(), [])
    
    out = grid.copy()
    out[out == 5] = 0
    if best_combo is not None:
        for r_top, c_left, placed in best_combo:
            for r, c in placed:
                out[r, c] = 1
                
    return out

if __name__ == "__main__":
    with open("training/6a1e5592.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_6a1e5592(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_6a1e5592(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 6a1e5592!")
