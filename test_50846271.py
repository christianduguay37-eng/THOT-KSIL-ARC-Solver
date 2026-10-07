import json
import numpy as np

def solve_50846271(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    
    pts2 = set(zip(*np.where(grid == 2)))
    if not pts2:
        return grid
        
    for R in [3, 2]:
        valid_centers = []
        for r0 in range(H):
            for c0 in range(W):
                cells = set()
                for dr in range(-R, R + 1):
                    if 0 <= r0 + dr < H:
                        cells.add((r0 + dr, c0))
                for dc in range(-R, R + 1):
                    if 0 <= c0 + dc < W:
                        cells.add((r0, c0 + dc))
                if all(grid[r, c] in (2, 5) for r, c in cells):
                    covered2 = cells & pts2
                    if len(covered2) >= 3:
                        valid_centers.append((r0, c0, cells, covered2))
                        
        valid_centers.sort(key=lambda x: len(x[3]), reverse=True)
        
        def find_cover(idx, current_covered, chosen):
            if current_covered == pts2:
                return chosen
            if idx >= len(valid_centers):
                return None
            r0, c0, cells, c2 = valid_centers[idx]
            overlap = False
            for _, _, chosen_cells in chosen:
                if cells & chosen_cells:
                    overlap = True
                    break
            if not overlap and (c2 - current_covered):
                res = find_cover(idx + 1, current_covered | c2, chosen + [(r0, c0, cells)])
                if res is not None:
                    return res
            return find_cover(idx + 1, current_covered, chosen)
            
        chosen_crosses = find_cover(0, set(), [])
        if chosen_crosses is not None:
            out = grid.copy()
            for r0, c0, cells in chosen_crosses:
                for r, c in cells:
                    if out[r, c] == 5:
                        out[r, c] = 8
            return out
            
    return grid

if __name__ == "__main__":
    with open("training/50846271.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_50846271(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_50846271(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 50846271!")
