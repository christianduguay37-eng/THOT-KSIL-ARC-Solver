import json
import numpy as np
from collections import Counter
from scipy.ndimage import label

def solve_57aa92db(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    
    labeled, num = label(grid != 0, structure=np.ones((3, 3)))
    
    comps = []
    for c_id in range(1, num + 1):
        pts = list(zip(*np.where(labeled == c_id)))
        colors = set(grid[r, c] for r, c in pts)
        comps.append((pts, colors))
        
    pair_colors = []
    for pts, colors in comps:
        if len(colors) == 2:
            pair_colors.append(colors)
            
    color_counts = Counter()
    for cols in pair_colors:
        for c in cols:
            color_counts[c] += 1
            
    if not color_counts:
        return grid
        
    anchor_color = color_counts.most_common(1)[0][0]
    
    template_comp = None
    target_comps = []
    
    for pts, colors in comps:
        if len(colors) == 2 and anchor_color in colors:
            shape_color = [c for c in colors if c != anchor_color][0]
            anchor_pts = [pt for pt in pts if grid[pt] == anchor_color]
            shape_pts = [pt for pt in pts if grid[pt] == shape_color]
            
            min_ar = min(r for r, c in anchor_pts)
            max_ar = max(r for r, c in anchor_pts)
            min_ac = min(c for r, c in anchor_pts)
            max_ac = max(c for r, c in anchor_pts)
            anc_h = max_ar - min_ar + 1
            anc_w = max_ac - min_ac + 1
            
            if anc_h == 1 and anc_w == 1 and len(shape_pts) >= 4:
                template_comp = (anchor_pts[0], shape_pts, shape_color)
            else:
                target_comps.append((anchor_pts, shape_pts, shape_color))
                
    if template_comp is None:
        return grid
        
    anc_pos, tpl_shape_pts, tpl_shape_color = template_comp
    rel_shape = [(r - anc_pos[0], c - anc_pos[1]) for r, c in tpl_shape_pts]
    
    out = grid.copy()
    
    for anc_pts, tgt_pts, tgt_color in target_comps:
        min_ar = min(r for r, c in anc_pts)
        max_ar = max(r for r, c in anc_pts)
        min_ac = min(c for r, c in anc_pts)
        max_ac = max(c for r, c in anc_pts)
        S = max_ar - min_ar + 1
        
        for dr, dc in rel_shape:
            block_r = min_ar + dr * S
            block_c = min_ac + dc * S
            for sr in range(S):
                for sc in range(S):
                    r = block_r + sr
                    c = block_c + sc
                    if 0 <= r < H and 0 <= c < W:
                        out[r, c] = tgt_color
                        
    return out

if __name__ == "__main__":
    with open("training/57aa92db.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_57aa92db(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_57aa92db(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 57aa92db!")
