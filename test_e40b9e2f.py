import json
import numpy as np

def solve_e40b9e2f(inp):
    H, W = inp.shape
    pts = [(r, c) for r in range(H) for c in range(W) if inp[r, c] != 0]
    if not pts: return inp
    cm_r = np.mean([r for r, c in pts])
    cm_c = np.mean([c for r, c in pts])
    
    # Candidate centers (step 0.5)
    candidates = []
    # Grid of centers around CM
    cr_vals = [cm_r + delta for delta in np.arange(-3.0, 3.5, 0.5)]
    # filter to half-integers
    cr_vals = [round(v * 2) / 2 for v in cr_vals]
    cc_vals = [round((cm_c + delta) * 2) / 2 for delta in np.arange(-3.0, 3.5, 0.5)]
    
    for cr in sorted(list(set(cr_vals))):
        for cc in sorted(list(set(cc_vals))):
            out = inp.copy()
            valid = True
            for r, c in pts:
                val = inp[r, c]
                dr, dc = r - cr, c - cc
                rotations = [
                    (cr + dr, cc + dc),
                    (cr - dc, cc + dr),
                    (cr - dr, cc - dc),
                    (cr + dc, cc - dr)
                ]
                for nr, nc in rotations:
                    if not (nr.is_integer() and nc.is_integer()):
                        valid = False; break
                    ir, ic = int(nr), int(nc)
                    if not (0 <= ir < H and 0 <= ic < W):
                        valid = False; break
                    if out[ir, ic] != 0 and out[ir, ic] != val:
                        valid = False; break
                    out[ir, ic] = val
                if not valid: break
            if valid:
                dist = (cr - cm_r)**2 + (cc - cm_c)**2
                candidates.append((dist, out))
                
    if not candidates: return inp
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

with open("training/e40b9e2f.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_e40b9e2f(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_e40b9e2f(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("e40b9e2f: 100% PASS ON ALL TRAIN AND TEST!")
