import json
import numpy as np

def solve_846bdb03(inp):
    H, W = inp.shape
    fours = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 4]
    fr_r0 = min(r for r, c in fours)
    fr_r1 = max(r for r, c in fours)
    fr_c0 = min(c for r, c in fours)
    fr_c1 = max(c for r, c in fours)
    
    H_out = fr_r1 - fr_r0 + 1
    W_out = fr_c1 - fr_c0 + 1
    
    left_color = inp[fr_r0 + 1, fr_c0]
    right_color = inp[fr_r0 + 1, fr_c1]
    
    out = np.zeros((H_out, W_out), dtype=int)
    out[0, 0] = 4
    out[0, -1] = 4
    out[-1, 0] = 4
    out[-1, -1] = 4
    
    out[1:-1, 0] = left_color
    out[1:-1, -1] = right_color
    
    inp_no_frame = inp.copy()
    inp_no_frame[fr_r0:fr_r1+1, fr_c0:fr_c1+1] = 0
    
    left_pts = [(r, c) for r in range(H) for c in range(W) if inp_no_frame[r, c] == left_color]
    if left_pts:
        lr_min = min(r for r, c in left_pts)
        lr_max = max(r for r, c in left_pts)
        lc_min = min(c for r, c in left_pts)
        lc_max = max(c for r, c in left_pts)
        shape_l = (inp_no_frame[lr_min:lr_max+1, lc_min:lc_max+1] == left_color)
        sh_h, sh_w = shape_l.shape
        for r in range(sh_h):
            for c in range(sh_w):
                if shape_l[r, c]:
                    out[1 + r, 1 + c] = left_color
                    
    right_pts = [(r, c) for r in range(H) for c in range(W) if inp_no_frame[r, c] == right_color]
    if right_pts:
        rr_min = min(r for r, c in right_pts)
        rr_max = max(r for r, c in right_pts)
        rc_min = min(c for r, c in right_pts)
        rc_max = max(c for r, c in right_pts)
        shape_r = (inp_no_frame[rr_min:rr_max+1, rc_min:rc_max+1] == right_color)
        sh_h, sh_w = shape_r.shape
        start_c = (W_out - 1) - sh_w
        for r in range(sh_h):
            for c in range(sh_w):
                if shape_r[r, c]:
                    out[1 + r, start_c + c] = right_color
                    
    return out

with open("training/846bdb03.json") as f:
    d = json.load(f)

p = d["train"][1]
inp = np.array(p["input"])
exp = np.array(p["output"])
pred = solve_846bdb03(inp)
print("EXP:\n", exp)
print("PRED:\n", pred)
print("Difference (exp - pred):\n", exp - pred)
