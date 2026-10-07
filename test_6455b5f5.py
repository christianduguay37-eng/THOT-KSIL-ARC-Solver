import json
import numpy as np
from scipy.ndimage import label

def solve_6455b5f5(grid):
    grid = np.array(grid, dtype=int)
    
    mask0 = (grid == 0)
    labeled, num = label(mask0, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    if num == 0:
        return grid
        
    sizes = {c_id: (labeled == c_id).sum() for c_id in range(1, num + 1)}
    max_size = max(sizes.values())
    min_size = min(sizes.values())
    
    out = grid.copy()
    for c_id, size in sizes.items():
        if size == max_size:
            out[labeled == c_id] = 1
        elif size == min_size:
            out[labeled == c_id] = 8
            
    return out

if __name__ == "__main__":
    with open("training/6455b5f5.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_6455b5f5(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_6455b5f5(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 6455b5f5!")
