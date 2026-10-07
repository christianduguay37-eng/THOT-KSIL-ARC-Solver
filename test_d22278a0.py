import json
import numpy as np

def get_corner_pattern(H, W, corner, color):
    pat = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            if corner == 'TL':
                r_tl, c_tl = r, c
            elif corner == 'TR':
                r_tl, c_tl = r, (W - 1) - c
            elif corner == 'BL':
                r_tl, c_tl = (H - 1) - r, c
            elif corner == 'BR':
                r_tl, c_tl = (H - 1) - r, (W - 1) - c
                
            if r_tl % 2 == 0:
                if c_tl <= r_tl or c_tl % 2 == 0:
                    pat[r, c] = color
            else:
                if c_tl > r_tl and c_tl % 2 == 0:
                    pat[r, c] = color
    return pat

def solve_d22278a0(grid):
    inp = np.array(grid)
    H, W = inp.shape
    
    corners = {}
    if inp[0, 0] != 0: corners['TL'] = ((0, 0), inp[0, 0])
    if inp[0, W-1] != 0: corners['TR'] = ((0, W-1), inp[0, W-1])
    if inp[H-1, 0] != 0: corners['BL'] = ((H-1, 0), inp[H-1, 0])
    if inp[H-1, W-1] != 0: corners['BR'] = ((H-1, W-1), inp[H-1, W-1])
    
    pats = {k: get_corner_pattern(H, W, k, v[1]) for k, v in corners.items()}
    
    pred = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            dists = {k: abs(r - v[0][0]) + abs(c - v[0][1]) for k, v in corners.items()}
            min_d = min(dists.values())
            closest = [k for k, d in dists.items() if d == min_d]
            if len(closest) == 1:
                pred[r, c] = pats[closest[0]][r, c]
            else:
                pred[r, c] = 0
                
    return pred.tolist()

if __name__ == "__main__":
    with open("training/d22278a0.json") as f:
        task = json.load(f)
    for idx, ex in enumerate(task["train"]):
        res = solve_d22278a0(ex["input"])
        assert res == ex["output"], f"Train {idx} failed!"
        print(f"Train {idx} PASS!")
    for idx, ex in enumerate(task["test"]):
        res = solve_d22278a0(ex["input"])
        print(f"Test {idx} output shape: {len(res)}x{len(res[0])}")
        print("Test 0 SUCCESS!")
