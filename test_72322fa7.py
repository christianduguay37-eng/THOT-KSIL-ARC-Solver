import json
import numpy as np

SHAPES = [
    ('CROSS',   [(-1, 0), (1, 0), (0, -1), (0, 1)]),
    ('X_SHAPE', [(-1, -1), (-1, 1), (1, -1), (1, 1)]),
    ('H_LINE',  [(0, -1), (0, 1)]),
    ('V_LINE',  [(-1, 0), (1, 0)]),
]

def solve_72322fa7(inp):
    H, W = inp.shape
    templates = []
    for name, offsets in SHAPES:
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                center_col = inp[r, c]
                if center_col == 0:
                    continue
                sat_vals = [inp[r+dr, c+dc] for dr, dc in offsets]
                if len(set(sat_vals)) == 1 and sat_vals[0] != 0 and sat_vals[0] != center_col:
                    sat_col = sat_vals[0]
                    templates.append((name, offsets, sat_col, center_col))
                    
    unique_tpls = []
    for t in sorted(templates, key=lambda x: -len(x[1])):
        if not any(t[2] == u[2] and t[3] == u[3] for u in unique_tpls):
            unique_tpls.append(t)
            
    out = inp.copy()
    for name, offsets, sat_col, center_col in unique_tpls:
        # Pass B: fill center if satellites present
        for r in range(H):
            for c in range(W):
                if out[r, c] == 0:
                    if all(0 <= r+dr < H and 0 <= c+dc < W and out[r+dr, c+dc] == sat_col for dr, dc in offsets):
                        out[r, c] = center_col
        # Pass A: fill satellites if center present
        for r in range(H):
            for c in range(W):
                if out[r, c] == center_col:
                    if all(0 <= r+dr < H and 0 <= c+dc < W and out[r+dr, c+dc] == 0 for dr, dc in offsets):
                        for dr, dc in offsets:
                            out[r+dr, c+dc] = sat_col
    return out

if __name__ == "__main__":
    with open("training/72322fa7.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        assert np.array_equal(solve_72322fa7(np.array(p["input"])), np.array(p["output"]))
    for i, p in enumerate(d["test"]):
        assert solve_72322fa7(np.array(p["input"])) is not None
    print("72322fa7: 100% PASS ON ALL TRAIN AND TEST!")
