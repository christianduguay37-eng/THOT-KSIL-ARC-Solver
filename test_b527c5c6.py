import json
import numpy as np
from scipy.ndimage import label

def solve_b527c5c6(inp):
    inp = np.array(inp)
    H, W = inp.shape
    out = inp.copy()
    
    lbl, num = label(inp != 0)
    bars = []
    for b in range(1, num + 1):
        r, c = np.where(lbl == b)
        h = r.max() - r.min() + 1
        w = c.max() - c.min() + 1
        r2, c2 = [(r_, c_) for r_, c_ in zip(r, c) if inp[r_, c_] == 2][0]
        kind = 'HORIZ' if w > h else 'VERT'
        bars.append({
            'kind': kind,
            'r_min': r.min(), 'r_max': r.max(),
            'c_min': c.min(), 'c_max': c.max(),
            'h': h, 'w': w,
            'r2': r2, 'c2': c2
        })
        
    horiz_bar = [b for b in bars if b['kind'] == 'HORIZ'][0]
    vert_bar = [b for b in bars if b['kind'] == 'VERT'][0]
    
    # 1. HORIZ bar shoots VERTICALLY from its nozzle 2
    if horiz_bar['r2'] == horiz_bar['r_min']:
        r_range = range(0, horiz_bar['r_min'])
    else:
        r_range = range(horiz_bar['r_max'] + 1, H)
        
    rad_h = horiz_bar['h'] - 1
    c2_h = horiz_bar['c2']
    for r in r_range:
        for c in range(c2_h - rad_h, c2_h + rad_h + 1):
            if 0 <= c < W:
                out[r, c] = 2 if c == c2_h else 3
                
    # 2. VERT bar shoots HORIZONTALLY from its nozzle 2
    if vert_bar['c2'] == vert_bar['c_min']:
        c_range = range(0, vert_bar['c_min'])
    else:
        c_range = range(vert_bar['c_max'] + 1, W)
        
    rad_v = vert_bar['w'] - 1
    r2_v = vert_bar['r2']
    for c in c_range:
        for r in range(r2_v - rad_v, r2_v + rad_v + 1):
            if 0 <= r < H:
                out[r, c] = 2 if r == r2_v else 3
                
    return out

if __name__ == "__main__":
    with open("training/b527c5c6.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_b527c5c6(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_b527c5c6(inp)
        if "output" in ex:
            out = np.array(ex["output"])
            if not np.array_equal(pred, out):
                print(f"FAILED Test {i}")
                all_ok = False
            else:
                print(f"PASS Test {i}")
        else:
            print(f"Test {i} output shape: {pred.shape}")
            
    if all_ok:
        print("ALL TESTS PASSED for b527c5c6!")
