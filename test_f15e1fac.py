import json
import numpy as np

def solve_f15e1fac(inp):
    H, W = inp.shape
    out = np.zeros_like(inp)
    
    # 8s are the beam sources, 2s are the deflectors
    eights = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 8]
    twos = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 2]
    
    # Keep the 2s in output
    for r, c in twos:
        out[r, c] = 2
        
    # Check direction of beams:
    # Are the 8s on the top edge (row 0)?
    if all(r == 0 for r, c in eights):
        # Beams go DOWN from row 0 to H-1
        # The 2s are on the left (col 0) or right (col W-1)
        two_cols = [c for r, c in twos]
        if 0 in two_cols:
            shift_dir = +1 # shift right
        else:
            shift_dir = -1 # shift left
            
        # Rows where deflectors exist
        two_rows = set(r for r, c in twos)
        
        current_cols = [c for r, c in eights]
        for r in range(H):
            if r in two_rows:
                current_cols = [c + shift_dir for c in current_cols]
            for c in current_cols:
                if 0 <= c < W:
                    out[r, c] = 8
                    
    # Are the 8s on the left edge (col 0)?
    elif all(c == 0 for r, c in eights):
        # Beams go RIGHT from col 0 to W-1
        two_rows = [r for r, c in twos]
        if 0 in two_rows:
            shift_dir = +1 # shift down
        else:
            shift_dir = -1 # shift up
            
        two_cols = set(c for r, c in twos)
        current_rows = [r for r, c in eights]
        for c in range(W):
            if c in two_cols:
                current_rows = [r + shift_dir for r in current_rows]
            for r in current_rows:
                if 0 <= r < H:
                    out[r, c] = 8
                    
    # Are the 8s on the right edge (col W-1)?
    elif all(c == W - 1 for r, c in eights):
        # Beams go LEFT from col W-1 down to 0
        two_rows = [r for r, c in twos]
        if 0 in two_rows:
            shift_dir = +1 # shift down
        else:
            shift_dir = -1 # shift up
            
        two_cols = set(c for r, c in twos)
        current_rows = [r for r, c in eights]
        for c in range(W - 1, -1, -1):
            if c in two_cols:
                current_rows = [r + shift_dir for r in current_rows]
            for r in current_rows:
                if 0 <= r < H:
                    out[r, c] = 8

    # Restore 2s in case overwritten
    for r, c in twos:
        out[r, c] = 2

    return out

with open("training/f15e1fac.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_f15e1fac(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_f15e1fac(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("f15e1fac: 100% PASS ON ALL TRAIN AND TEST!")
