import json
import numpy as np

def solve_8eb1be9a(inp):
    H, W = inp.shape
    non_empty = [r for r in range(H) if np.any(inp[r] != 0)]
    r0 = min(non_empty)
    out = np.zeros_like(inp)
    for r in range(H):
        src_r = r0 + ((r - r0) % 3)
        out[r, :] = inp[src_r, :]
    return out

with open("training/8eb1be9a.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_8eb1be9a(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_8eb1be9a(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("8eb1be9a: 100% PASS ON ALL TRAIN AND TEST!")
