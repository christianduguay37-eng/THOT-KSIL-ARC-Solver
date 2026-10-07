import json
import numpy as np

def solve_ae3edfdc(inp):
    H, W = inp.shape
    out = np.zeros_like(inp)
    
    # Centers are 1 and 2
    centers = [(r, c, inp[r, c]) for r in range(H) for c in range(W) if inp[r, c] in (1, 2)]
    # Keep centers
    for r, c, val in centers:
        out[r, c] = val
        
    # Satellites are 3 and 7
    satellites = [(r, c, inp[r, c]) for r in range(H) for c in range(W) if inp[r, c] in (3, 7)]
    
    for sr, sc, sval in satellites:
        # Find which center it aligns with (same row or same col)
        for cr, cc, cval in centers:
            if sr == cr:
                # Same row
                if sc < cc:
                    out[cr, cc - 1] = sval
                else:
                    out[cr, cc + 1] = sval
            elif sc == cc:
                # Same col
                if sr < cr:
                    out[cr - 1, cc] = sval
                else:
                    out[cr + 1, cc] = sval
    return out

with open("training/ae3edfdc.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_ae3edfdc(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_ae3edfdc(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("ae3edfdc: 100% PASS ON ALL TRAIN AND TEST!")
