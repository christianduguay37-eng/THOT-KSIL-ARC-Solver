import json
import numpy as np
from scipy.ndimage import label

def solve_db93a21d(grid):
    inp = np.array(grid)
    GH, GW = inp.shape
    labeled, num = label(inp == 9)
    res = np.zeros((GH, GW), dtype=int)
    
    blocks = []
    for b in range(1, num + 1):
        rows, cols = np.where(labeled == b)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        w = c1 - c0
        h = r1 - r0
        t = max(w, h) // 2
        blocks.append((r0, r1, c0, c1, t))
        
    # Step 1: Draw blue (1) shadows from r1 down to bottom edge
    for r0, r1, c0, c1, t in blocks:
        for r in range(r1, GH):
            for c in range(c0, c1):
                res[r, c] = 1
                
    # Step 2: Draw green (3) frames around each block
    for r0, r1, c0, c1, t in blocks:
        for r in range(max(0, r0 - t), min(GH, r1 + t)):
            for c in range(max(0, c0 - t), min(GW, c1 + t)):
                res[r, c] = 3
                
    # Step 3: Draw original maroon (9) blocks
    for r0, r1, c0, c1, t in blocks:
        res[r0:r1, c0:c1] = 9
        
    return res.tolist()

if __name__ == "__main__":
    with open("training/db93a21d.json") as f:
        task = json.load(f)
    for idx, ex in enumerate(task["train"]):
        res = solve_db93a21d(ex["input"])
        assert res == ex["output"], f"Train {idx} failed!"
        print(f"Train {idx} PASS!")
    for idx, ex in enumerate(task["test"]):
        res = solve_db93a21d(ex["input"])
        print(f"Test {idx} output shape: {len(res)}x{len(res[0])}")
        print("Test 0 SUCCESS!")
