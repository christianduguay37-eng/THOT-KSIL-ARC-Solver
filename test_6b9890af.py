import json
import numpy as np

def solve_6b9890af(inp):
    twos = np.argwhere(inp == 2)
    r0, c0 = twos.min(axis=0)
    r1, c1 = twos.max(axis=0)
    S = r1 - r0 + 1
    
    fg = [c for c in np.unique(inp) if c not in (0, 2)][0]
    pts = np.argwhere(inp == fg)
    fr0, fc0 = pts.min(axis=0)
    fr1, fc1 = pts.max(axis=0)
    pattern_3x3 = inp[fr0:fr1+1, fc0:fc1+1]
    
    inner_size = S - 2
    k = inner_size // 3
    scaled = np.kron(pattern_3x3, np.ones((k, k), dtype=int))
    
    out = np.zeros((S, S), dtype=int)
    out[0, :] = 2
    out[-1, :] = 2
    out[:, 0] = 2
    out[:, -1] = 2
    out[1:-1, 1:-1] = scaled
    return out

with open("training/6b9890af.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_6b9890af(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_6b9890af(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("6b9890af: 100% PASS ON ALL TRAIN AND TEST!")
