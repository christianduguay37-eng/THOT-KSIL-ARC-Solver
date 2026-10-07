import json
import numpy as np

def solve_846bdb03(inp):
    pts4 = np.argwhere(inp == 4)
    if len(pts4) < 4:
        return None
    r_min_4, c_min_4 = pts4.min(0)
    r_max_4, c_max_4 = pts4.max(0)
    c_left = inp[r_min_4 + 1, c_min_4]
    c_right = inp[r_min_4 + 1, c_max_4]
    
    outside_mask = np.ones(inp.shape, dtype=bool)
    outside_mask[r_min_4:r_max_4+1, c_min_4:c_max_4+1] = False
    outside = inp * outside_mask
    
    pts_obj = np.argwhere((outside == c_left) | (outside == c_right))
    if len(pts_obj) == 0:
        return None
    r0, c0 = pts_obj.min(0)
    r1, c1 = pts_obj.max(0)
    bicolor_obj = outside[r0:r1+1, c0:c1+1].copy()
    
    cols_left = np.argwhere(bicolor_obj == c_left)[:, 1]
    cols_right = np.argwhere(bicolor_obj == c_right)[:, 1]
    if len(cols_left) == 0 or len(cols_right) == 0:
        return None
    if cols_left.mean() > cols_right.mean():
        bicolor_obj = np.fliplr(bicolor_obj)
        
    H_obj, W_obj = bicolor_obj.shape
    pred = np.zeros((H_obj + 2, W_obj + 2), dtype=int)
    pred[0, 0] = 4
    pred[0, -1] = 4
    pred[-1, 0] = 4
    pred[-1, -1] = 4
    pred[1:-1, 0] = c_left
    pred[1:-1, -1] = c_right
    pred[1:-1, 1:-1] = bicolor_obj
    return pred

if __name__ == "__main__":
    with open("training/846bdb03.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        assert np.array_equal(solve_846bdb03(np.array(p["input"])), np.array(p["output"]))
    for i, p in enumerate(d["test"]):
        assert solve_846bdb03(np.array(p["input"])) is not None
    print("846bdb03: 100% PASS ON ALL TRAIN AND TEST!")
