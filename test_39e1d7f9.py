import json
import numpy as np

def solve_39e1d7f9(inp):
    c_wall = np.argmax(np.bincount(inp.flatten())[1:]) + 1
    H, W = inp.shape
    row_lines = [-1] + [r for r in range(H) if (inp[r] == c_wall).sum() > W * 0.7] + [H]
    col_lines = [-1] + [c for c in range(W) if (inp[:, c] == c_wall).sum() > H * 0.7] + [W]
    
    NR = len(row_lines) - 1
    NC = len(col_lines) - 1
    grid_in = np.zeros((NR, NC), dtype=int)
    for i in range(NR):
        for j in range(NC):
            r0, r1 = row_lines[i] + 1, row_lines[i+1]
            c0, c1 = col_lines[j] + 1, col_lines[j+1]
            grid_in[i, j] = int(np.median(inp[r0:r1, c0:c1]))
            
    pattern = []
    center_color = None
    center_pos = None
    for r in range(NR):
        for c in range(NC):
            val = grid_in[r, c]
            if val != 0:
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < NR and 0 <= nc < NC and grid_in[nr, nc] != 0:
                            neighbors.append((dr, dc, grid_in[nr, nc]))
                if len(neighbors) > len(pattern):
                    pattern = neighbors
                    center_color = val
                    center_pos = (r, c)
                    
    grid_out = grid_in.copy()
    for r in range(NR):
        for c in range(NC):
            if grid_in[r, c] == center_color:
                for dr, dc, col in pattern:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < NR and 0 <= nc < NC:
                        if grid_out[nr, nc] == 0:
                            grid_out[nr, nc] = col
                            
    out = inp.copy()
    for i in range(NR):
        for j in range(NC):
            r0, r1 = row_lines[i] + 1, row_lines[i+1]
            c0, c1 = col_lines[j] + 1, col_lines[j+1]
            val = grid_out[i, j]
            if val != 0:
                out[r0:r1, c0:c1] = val
    return out

with open("training/39e1d7f9.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_39e1d7f9(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_39e1d7f9(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("39e1d7f9: 100% PASS ON ALL TRAIN AND TEST!")
