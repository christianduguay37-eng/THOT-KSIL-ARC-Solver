import json
import numpy as np
from scipy.ndimage import label

def solve_3e980e27(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    
    labeled, num = label(grid != 0, structure=np.ones((3, 3)))
    templates = {}
    dots = []
    
    for c_id in range(1, num + 1):
        pts = list(zip(*np.where(labeled == c_id)))
        if len(pts) == 1:
            dots.append((pts[0], grid[pts[0]]))
        else:
            anchors = [pt for pt in pts if grid[pt] in (2, 3)]
            if len(anchors) == 1:
                anc = anchors[0]
                anc_color = grid[anc]
                other_pts = [(r - anc[0], c - anc[1], grid[r, c]) for r, c in pts if (r, c) != anc]
                templates[anc_color] = other_pts
                
    out = grid.copy()
    for dot_pos, dot_color in dots:
        if dot_color not in templates:
            continue
        tpl_others = templates[dot_color]
        for dr, dc, col in tpl_others:
            if dot_color == 2:
                # fliplr
                nr, nc = dr, -dc
            else:
                # id
                nr, nc = dr, dc
            tr = dot_pos[0] + nr
            tc = dot_pos[1] + nc
            if 0 <= tr < H and 0 <= tc < W:
                out[tr, tc] = col
                
    return out

if __name__ == "__main__":
    with open("training/3e980e27.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_3e980e27(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_3e980e27(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 3e980e27!")
