import json
import numpy as np

def solve_ecdecbb3(inp):
    H, W = inp.shape
    out = inp.copy()
    
    eights = np.argwhere(inp == 8)
    row_counts = [(inp[r] == 8).sum() for r in range(H)]
    col_counts = [(inp[:, c] == 8).sum() for c in range(W)]
    
    is_horizontal = (max(row_counts) > max(col_counts))
    twos = np.argwhere(inp == 2)
    
    if is_horizontal:
        lines_r = [r for r in range(H) if (inp[r] == 8).sum() >= W * 0.7]
        for tr, tc in twos:
            top_lines = [lr for lr in lines_r if lr < tr]
            bot_lines = [lr for lr in lines_r if lr > tr]
            if top_lines and bot_lines:
                targets = [max(top_lines), min(bot_lines)]
            elif top_lines:
                targets = [max(top_lines)]
            elif bot_lines:
                targets = [min(bot_lines)]
            else:
                targets = []
                
            for target_r in targets:
                r_start = min(tr, target_r)
                r_end = max(tr, target_r)
                out[r_start:r_end+1, tc] = 2
                r0 = max(0, target_r - 1)
                r1 = min(H, target_r + 2)
                c0 = max(0, tc - 1)
                c1 = min(W, tc + 2)
                out[r0:r1, c0:c1] = 8
                out[target_r, tc] = 2
    else:
        lines_c = [c for c in range(W) if (inp[:, c] == 8).sum() >= H * 0.7]
        for tr, tc in twos:
            left_lines = [lc for lc in lines_c if lc < tc]
            right_lines = [lc for lc in lines_c if lc > tc]
            if left_lines and right_lines:
                targets = [max(left_lines), min(right_lines)]
            elif left_lines:
                targets = [max(left_lines)]
            elif right_lines:
                targets = [min(right_lines)]
            else:
                targets = []
                
            for target_c in targets:
                c_start = min(tc, target_c)
                c_end = max(tc, target_c)
                out[tr, c_start:c_end+1] = 2
                r0 = max(0, tr - 1)
                r1 = min(H, tr + 2)
                c0 = max(0, target_c - 1)
                c1 = min(W, target_c + 2)
                out[r0:r1, c0:c1] = 8
                out[tr, target_c] = 2
                
    return out

with open("training/ecdecbb3.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_ecdecbb3(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_ecdecbb3(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("ecdecbb3: 100% PASS ON ALL TRAIN AND TEST!")
