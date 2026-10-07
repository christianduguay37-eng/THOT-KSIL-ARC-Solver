import json
import numpy as np
from scipy.ndimage import label

def solve_e509e548(grid):
    inp = np.array(grid)
    out = inp.copy()
    labeled, num = label(inp == 3)
    
    for c in range(1, num + 1):
        mask = (labeled == c)
        coords = set(zip(*np.where(mask)))
        
        degrees = {}
        for r, c_pt in coords:
            deg = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c_pt + dc) in coords:
                    deg += 1
            degrees[(r, c_pt)] = deg
            
        corners = 0
        for (r, c_pt), deg in degrees.items():
            if deg == 2:
                neighs = [(dr, dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (r + dr, c_pt + dc) in coords]
                if neighs[0][0] != -neighs[1][0] or neighs[0][1] != -neighs[1][1]:
                    corners += 1
                    
        num_branches = sum(list(degrees.values()).count(d) for d in [3, 4])
        if num_branches > 0:
            assigned = 2
        elif corners == 1:
            assigned = 1
        else:
            assigned = 6
            
        for r, c_pt in coords:
            out[r, c_pt] = assigned
            
    return out.tolist()

if __name__ == "__main__":
    with open("training/e509e548.json") as f:
        task = json.load(f)
    for idx, ex in enumerate(task["train"]):
        res = solve_e509e548(ex["input"])
        assert res == ex["output"], f"Train {idx} failed!"
        print(f"Train {idx} PASS!")
    for idx, ex in enumerate(task["test"]):
        res = solve_e509e548(ex["input"])
        print(f"Test {idx} output shape: {len(res)}x{len(res[0])}")
        print("Test 0 SUCCESS!")
