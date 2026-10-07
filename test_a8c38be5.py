import json
import numpy as np

def solve_a8c38be5(inp):
    H, W = inp.shape
    pieces = []
    visited_cells = set()
    for r in range(H - 2):
        for c in range(W - 2):
            sub = inp[r:r+3, c:c+3]
            if np.all(sub != 0):
                cells = {(r+dr, c+dc) for dr in range(3) for dc in range(3)}
                if not (cells & visited_cells):
                    pieces.append(sub)
                    visited_cells.update(cells)
                    
    out = np.zeros((9, 9), dtype=int)
    for p in pieces:
        top_5 = np.all(p[0, :] == 5)
        bot_5 = np.all(p[2, :] == 5)
        left_5 = np.all(p[:, 0] == 5)
        right_5 = np.all(p[:, 2] == 5)
        
        if np.all(p == 5):
            br, bc = 1, 1
        elif bot_5 and right_5:
            br, bc = 0, 0
        elif bot_5 and left_5:
            br, bc = 0, 2
        elif top_5 and right_5:
            br, bc = 2, 0
        elif top_5 and left_5:
            br, bc = 2, 2
        elif bot_5:
            br, bc = 0, 1
        elif top_5:
            br, bc = 2, 1
        elif right_5:
            br, bc = 1, 0
        elif left_5:
            br, bc = 1, 2
        else:
            raise ValueError("Unclassified piece")
        out[br*3:(br+1)*3, bc*3:(bc+1)*3] = p
    return out

with open("training/a8c38be5.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_a8c38be5(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_a8c38be5(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("a8c38be5: 100% PASS ON ALL TRAIN AND TEST!")
