import json
import numpy as np
from scipy.ndimage import label

def solve_b775ac94(inp):
    inp = np.array(inp)
    H, W = inp.shape
    out = inp.copy()
    
    shapes = []
    for c in np.unique(inp):
        if c == 0:
            continue
        lbl, num = label(inp == c, structure=np.ones((3, 3)))
        for comp_id in range(1, num + 1):
            coords = np.argwhere(lbl == comp_id)
            if len(coords) > 1:
                shapes.append((c, set(map(tuple, coords))))
                
    quads = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for r0 in range(H - 1):
        for c0 in range(W - 1):
            w = inp[r0:r0+2, c0:c0+2]
            if np.sum(w != 0) < 2:
                continue
            if len(np.unique(w[w != 0])) < 2:
                continue
                
            for qr, qc in quads:
                p = (r0 + qr, c0 + qc)
                matching = [s for s in shapes if p in s[1]]
                if not matching:
                    continue
                c_main, shape_coords = matching[0]
                
                r_ok = all(r <= r0 for r, c in shape_coords) if qr == 0 else all(r >= r0 + 1 for r, c in shape_coords)
                c_ok = all(c <= c0 for r, c in shape_coords) if qc == 0 else all(c >= c0 + 1 for r, c in shape_coords)
                if not (r_ok and c_ok):
                    continue
                    
                for tr, tc in quads:
                    if (tr, tc) == (qr, qc):
                        continue
                    c_target = w[tr, tc]
                    if c_target == 0:
                        continue
                    for r, c in shape_coords:
                        nr = (2 * r0 + 1 - r) if tr != qr else r
                        nc = (2 * c0 + 1 - c) if tc != qc else c
                        if 0 <= nr < H and 0 <= nc < W:
                            out[nr, nc] = c_target
                            
    return out

if __name__ == "__main__":
    with open("training/b775ac94.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_b775ac94(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_b775ac94(inp)
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
        print("ALL TESTS PASSED for b775ac94!")
