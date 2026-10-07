import json
import numpy as np
from scipy.ndimage import label

def solve_fcb5c309(inp):
    colors = [c for c in np.unique(inp) if c != 0]
    assert len(colors) == 2, f"Expected 2 colors, got {len(colors)}"
    
    # Determine which color is the box color (has the largest single connected component)
    max_sizes = {}
    for c in colors:
        lab, num = label(inp == c)
        max_sizes[c] = max([np.sum(lab == i) for i in range(1, num + 1)])
        
    box_col = max(colors, key=lambda c: max_sizes[c])
    dot_col = [c for c in colors if c != box_col][0]
    
    # Find all components of box_col
    lab, num = label(inp == box_col)
    
    best_box = None
    max_dots_inside = -1
    max_area = -1
    
    for feat in range(1, num + 1):
        pts = [(r, c) for r in range(inp.shape[0]) for c in range(inp.shape[1]) if lab[r, c] == feat]
        r0, r1 = min(r for r, c in pts), max(r for r, c in pts)
        c0, c1 = min(c for r, c in pts), max(c for r, c in pts)
        
        # count dots inside strictly interior
        dots_inside = sum(1 for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if inp[r, c] == dot_col)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        
        if dots_inside > max_dots_inside or (dots_inside == max_dots_inside and area > max_area):
            max_dots_inside = dots_inside
            max_area = area
            best_box = (r0, r1, c0, c1)
            
    r0, r1, c0, c1 = best_box
    crop = inp[r0:r1+1, c0:c1+1]
    
    # In the crop: box_col becomes dot_col, dot_col stays dot_col, 0 stays 0
    out = np.where((crop == box_col) | (crop == dot_col), dot_col, 0)
    return out

with open("training/fcb5c309.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_fcb5c309(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_fcb5c309(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("fcb5c309: 100% PASS ON ALL TRAIN AND TEST!")
