import json
import numpy as np
from scipy.ndimage import label

def solve_776ffc46(inp):
    H, W = inp.shape
    box = None
    for r0 in range(H):
        for c0 in range(W):
            if inp[r0, c0] != 5:
                continue
            for r1 in range(r0 + 4, H):
                if inp[r1, c0] != 5:
                    break
                for c1 in range(c0 + 4, W):
                    if inp[r0, c1] != 5 or inp[r1, c1] != 5:
                        continue
                    if not np.all(inp[r0, c0:c1+1] == 5):
                        continue
                    if not np.all(inp[r1, c0:c1+1] == 5):
                        continue
                    if not np.all(inp[r0:r1+1, c0] == 5):
                        continue
                    if not np.all(inp[r0:r1+1, c1] == 5):
                        continue
                    inside = inp[r0+1:r1, c0+1:c1]
                    non_zeros = inside[inside != 0]
                    if len(non_zeros) > 0 and 5 not in non_zeros:
                        box = (r0, c0, r1, c1, inside)
                        break
                if box:
                    break
            if box:
                break
        if box:
            break
        
    if not box:
        return None
    r0, c0, r1, c1, inside = box
    c_target = [c for c in np.unique(inside) if c != 0][0]
    pts_tpl = np.argwhere(inside == c_target)
    tpl_norm = set(map(tuple, pts_tpl - pts_tpl.min(axis=0)))
    
    out = inp.copy()
    mask1 = (inp == 1)
    lbl, num = label(mask1)
    for i in range(1, num + 1):
        pts = np.argwhere(lbl == i)
        pts_norm = set(map(tuple, pts - pts.min(axis=0)))
        if pts_norm == tpl_norm:
            for r, c in pts:
                out[r, c] = c_target
    return out

if __name__ == "__main__":
    with open("training/776ffc46.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        assert np.array_equal(solve_776ffc46(np.array(p["input"])), np.array(p["output"]))
    for i, p in enumerate(d["test"]):
        assert solve_776ffc46(np.array(p["input"])) is not None
    print("776ffc46: 100% PASS ON ALL TRAIN AND TEST!")
