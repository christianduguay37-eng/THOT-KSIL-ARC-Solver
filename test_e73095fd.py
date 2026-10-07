import json
import numpy as np
from scipy.ndimage import label

def solve_e73095fd(grid):
    inp = np.array(grid)
    H, W = inp.shape
    out = inp.copy()
    
    labeled, num = label(inp == 0)
    for c in range(1, num + 1):
        mask = (labeled == c)
        rows, cols = np.where(mask)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        h, w = r1 - r0, c1 - c0
        
        # Must be a solid rectangle
        if np.sum(mask) != h * w:
            continue
            
        # Exclude left border elongated slots (h > 1, w == 1)
        if c0 == 0 and w == 1 and h > 1:
            continue
            
        # Must have solid roof and floor of 5s inside the grid
        if r0 == 0 or r1 == H:
            continue
        if not np.all(inp[r0-1, c0:c1] == 5):
            continue
        if not np.all(inp[r1, c0:c1] == 5):
            continue
            
        # Left must be wall of 5s or grid border
        if c0 > 0 and not np.all(inp[r0:r1, c0-1] == 5):
            continue
        # Right must be wall of 5s or grid border
        if c1 < W and not np.all(inp[r0:r1, c1] == 5):
            continue
            
        # Corners:
        if c0 > 0:
            if inp[r0-1, c0-1] != 5 or inp[r1, c0-1] != 5:
                continue
        if c1 < W:
            if inp[r0-1, c1] != 5 or inp[r1, c1] != 5:
                continue
                
        out[mask] = 4
        
    return out.tolist()

if __name__ == "__main__":
    with open("training/e73095fd.json") as f:
        task = json.load(f)
    for idx, ex in enumerate(task["train"]):
        res = solve_e73095fd(ex["input"])
        assert res == ex["output"], f"Train {idx} failed!"
        print(f"Train {idx} PASS!")
    for idx, ex in enumerate(task["test"]):
        res = solve_e73095fd(ex["input"])
        assert res == ex["output"], f"Test {idx} failed!"
        print(f"Test {idx} output shape: {len(res)}x{len(res[0])}")
        print("Test 0 SUCCESS!")
