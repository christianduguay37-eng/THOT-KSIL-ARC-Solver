import json
import numpy as np

def solve_10fcaaa3(inp):
    H, W = inp.shape
    out = np.tile(inp, (2, 2))
    dots = np.argwhere(out != 0)
    for r, c in dots:
        for dr in (-1, 1):
            for dc in (-1, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 2*H and 0 <= nc < 2*W:
                    if out[nr, nc] == 0:
                        out[nr, nc] = 8
    return out

if __name__ == "__main__":
    with open("training/10fcaaa3.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        assert np.array_equal(solve_10fcaaa3(np.array(p["input"])), np.array(p["output"]))
    for i, p in enumerate(d["test"]):
        assert np.array_equal(solve_10fcaaa3(np.array(p["input"])), np.array(p["output"]))
    print("10fcaaa3: 100% PASS ON ALL TRAIN AND TEST!")
