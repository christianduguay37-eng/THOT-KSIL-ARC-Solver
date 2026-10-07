import json
import numpy as np
from scipy.ndimage import label

def get_d4_transforms():
    return [
        lambda r, c: (r, c),
        lambda r, c: (c, -r),
        lambda r, c: (-r, -c),
        lambda r, c: (-c, r),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (c, r),
        lambda r, c: (-c, -r)
    ]

def solve_36d67576(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    
    mask4 = (grid == 4)
    labeled, num_comp = label(mask4, structure=np.ones((3, 3)))
    
    comps = []
    template_idx = -1
    
    for c_id in range(1, num_comp + 1):
        pts4 = set(zip(*np.where(labeled == c_id)))
        
        twos = []
        others = []
        for r, c in pts4:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        val = grid[nr, nc]
                        if val == 2 and (nr, nc) not in twos:
                            twos.append((nr, nc))
                        elif val in (1, 3) and (nr, nc, val) not in others:
                            others.append((nr, nc, val))
                            
        comp_info = {
            'pts4': pts4,
            'twos': twos,
            'others': others
        }
        comps.append(comp_info)
        if len(others) > 0 and len(twos) > 0:
            template_idx = len(comps) - 1
            
    if template_idx == -1:
        return grid
        
    tpl = comps[template_idx]
    tpl_pts4 = tpl['pts4']
    if not tpl['twos']:
        return grid
    tpl_anchor = tpl['twos'][0]
    tpl_others = tpl['others']
    
    origin_tpl = min(tpl_pts4)
    tpl_pts4_rel = set((r - origin_tpl[0], c - origin_tpl[1]) for r, c in tpl_pts4)
    tpl_anchor_rel = (tpl_anchor[0] - origin_tpl[0], tpl_anchor[1] - origin_tpl[1])
    tpl_others_rel = [(r - origin_tpl[0], c - origin_tpl[1], col) for r, c, col in tpl_others]
    
    d4 = get_d4_transforms()
    out = grid.copy()
    
    for idx, comp in enumerate(comps):
        if idx == template_idx:
            continue
        tgt_pts4 = comp['pts4']
        if not comp['twos']:
            continue
        tgt_anchor = comp['twos'][0]
        
        for f in d4:
            t_pts4 = set(f(r, c) for r, c in tpl_pts4_rel)
            t_anchor = f(*tpl_anchor_rel)
            
            tr = tgt_anchor[0] - t_anchor[0]
            tc = tgt_anchor[1] - t_anchor[1]
            
            translated_pts4 = set((r + tr, c + tc) for r, c in t_pts4)
            if translated_pts4 == tgt_pts4:
                for dr, dc, col in tpl_others_rel:
                    fr, fc = f(dr, dc)
                    out_r = fr + tr
                    out_c = fc + tc
                    if 0 <= out_r < H and 0 <= out_c < W:
                        out[out_r, out_c] = col
                break
                
    return out

if __name__ == "__main__":
    with open("training/36d67576.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_36d67576(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_36d67576(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 36d67576!")
