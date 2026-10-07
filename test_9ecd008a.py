import json
import numpy as np

def solve_9ecd008a(inp):
    zeros = np.argwhere(inp == 0)
    zr, zc = zeros.min(axis=0)
    
    # 16x16 grid has D4 symmetry
    # Try horizontal reflection: (r, 15 - c)
    cand_h = np.fliplr(inp)[zr:zr+3, zc:zc+3]
    if (cand_h != 0).all():
        return cand_h
        
    # Try vertical reflection: (15 - r, c)
    cand_v = np.flipud(inp)[zr:zr+3, zc:zc+3]
    if (cand_v != 0).all():
        return cand_v
        
    # Try transpose: (c, r)
    cand_t = inp.T[zr:zr+3, zc:zc+3]
    if (cand_t != 0).all():
        return cand_t
        
    # Try rot180
    cand_r180 = np.rot90(inp, 2)[zr:zr+3, zc:zc+3]
    return cand_r180

with open("training/9ecd008a.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_9ecd008a(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_9ecd008a(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("9ecd008a: 100% PASS ON ALL TRAIN AND TEST!")
