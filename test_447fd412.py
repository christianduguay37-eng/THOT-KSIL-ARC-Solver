import json
import numpy as np
from scipy.ndimage import label

def solve_447fd412(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    
    labeled_all, num_all = label((grid == 1) | (grid == 2), structure=np.ones((3, 3)))
    
    template_id = -1
    for c_id in range(1, num_all + 1):
        c_mask = (labeled_all == c_id)
        if (grid[c_mask] == 1).any() and (grid[c_mask] == 2).any():
            template_id = c_id
            break
            
    if template_id == -1:
        return grid
        
    tpl_mask = (labeled_all == template_id)
    pts1_tpl = list(zip(*np.where(tpl_mask & (grid == 1))))
    
    tgt_mask = (grid == 2) & (~tpl_mask)
    labeled_tgt, num_tgt = label(tgt_mask, structure=np.ones((3, 3)))
    
    labeled_tpl2, num_tpl2 = label(tpl_mask & (grid == 2), structure=np.ones((3, 3)))
    tpl_blocks = []
    for b_id in range(1, num_tpl2 + 1):
        b_pts = list(zip(*np.where(labeled_tpl2 == b_id)))
        min_r = min(r for r, c in b_pts)
        min_c = min(c for r, c in b_pts)
        tpl_blocks.append((min_r, min_c))
    tpl_blocks.sort()
    
    if not tpl_blocks:
        return grid
        
    anc0_r, anc0_c = tpl_blocks[0]
    rel_1s = [(r - anc0_r, c - anc0_c) for r, c in pts1_tpl]
    
    out = grid.copy()
    
    tgt_blocks = []
    for t_id in range(1, num_tgt + 1):
        t_pts = list(zip(*np.where(labeled_tgt == t_id)))
        min_r = min(r for r, c in t_pts)
        max_r = max(r for r, c in t_pts)
        min_c = min(c for r, c in t_pts)
        max_c = max(c for r, c in t_pts)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        S = max(h, w)
        tgt_blocks.append((min_r, min_c, S))
        
    tgt_blocks.sort()
    
    if num_tpl2 == 1:
        for min_r, min_c, S in tgt_blocks:
            for dr, dc in rel_1s:
                for sr in range(S):
                    for sc in range(S):
                        out_r = min_r + dr * S + sr
                        out_c = min_c + dc * S + sc
                        if 0 <= out_r < H and 0 <= out_c < W and out[out_r, out_c] == 0:
                            out[out_r, out_c] = 1
    elif num_tpl2 == 2:
        anc1_r, anc1_c = tpl_blocks[1]
        tpl_dr = anc1_r - anc0_r
        tpl_dc = anc1_c - anc0_c
        
        used = set()
        for i in range(len(tgt_blocks)):
            if i in used:
                continue
            r1, c1, S1 = tgt_blocks[i]
            for j in range(i + 1, len(tgt_blocks)):
                if j in used:
                    continue
                r2, c2, S2 = tgt_blocks[j]
                if S1 == S2:
                    S = S1
                    if r2 - r1 == tpl_dr * S and c2 - c1 == tpl_dc * S:
                        used.add(i)
                        used.add(j)
                        for dr, dc in rel_1s:
                            for sr in range(S):
                                for sc in range(S):
                                    out_r = r1 + dr * S + sr
                                    out_c = c1 + dc * S + sc
                                    if 0 <= out_r < H and 0 <= out_c < W and out[out_r, out_c] == 0:
                                        out[out_r, out_c] = 1
                        break
                        
    return out

if __name__ == "__main__":
    with open("training/447fd412.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_447fd412(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_447fd412(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 447fd412!")
