import json
import numpy as np

def solve_539a4f51(inp):
    # Nonzeros define K
    rows = [r for r in range(inp.shape[0]) if np.any(inp[r, :] != 0)]
    cols = [c for c in range(inp.shape[1]) if np.any(inp[:, c] != 0)]
    K = max(max(rows), max(cols)) + 1
    B = inp[:K, :K]
    bg = B[0, 0]
    out = np.full((10, 10), bg)
    # top-left: B
    out[:K, :K] = B
    # top-right: tile(B[0, :])
    out[:K, K:2*K] = np.tile(B[0, :], (K, 1))
    # bot-left: tile(B[:, 0])
    out[K:2*K, :K] = np.tile(B[:, 0][:, None], (1, K))
    # bot-right: B
    out[K:2*K, K:2*K] = B
    return out

with open("training/539a4f51.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_539a4f51(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_539a4f51(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("539a4f51: 100% PASS ON ALL TRAIN AND TEST!")
