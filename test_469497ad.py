import json
import numpy as np

def solve_469497ad(inp):
    # Border is row 4 and col 4
    border_colors = set(inp[4, :].tolist() + inp[:, 4].tolist()) - {0}
    k = len(border_colors) + 1
    
    # Scale the input grid by k (Kronecker product)
    out = np.kron(inp, np.ones((k, k), dtype=int))
    H, W = out.shape
    
    # Find the 2x2 block of non-zero, non-border color in inp[:4, :4]
    inner = inp[:4, :4]
    block_color = None
    block_r0, block_c0 = None, None
    for r in range(3):
        for c in range(3):
            if inner[r, c] != 0 and inner[r, c] not in border_colors:
                if (inner[r:r+2, c:c+2] == inner[r, c]).all():
                    block_color = inner[r, c]
                    block_r0, block_c0 = r, c
                    break
        if block_color is not None:
            break
            
    # In out, the block is from r_start to r_end, c_start to c_end
    r_start = block_r0 * k
    r_end = (block_r0 + 2) * k - 1
    c_start = block_c0 * k
    c_end = (block_c0 + 2) * k - 1
    
    # Diagonal rays of 2s shoot from the 4 corners into the zero region
    # 1. Top-Left: from (r_start - 1, c_start - 1) stepping (-1, -1)
    r, c = r_start - 1, c_start - 1
    while r >= 0 and c >= 0 and out[r, c] == 0:
        out[r, c] = 2
        r -= 1
        c -= 1
        
    # 2. Top-Right: from (r_start - 1, c_end + 1) stepping (-1, +1)
    r, c = r_start - 1, c_end + 1
    while r >= 0 and c < 4 * k and out[r, c] == 0:
        out[r, c] = 2
        r -= 1
        c += 1
        
    # 3. Bottom-Left: from (r_end + 1, c_start - 1) stepping (+1, -1)
    r, c = r_end + 1, c_start - 1
    while r < 4 * k and c >= 0 and out[r, c] == 0:
        out[r, c] = 2
        r += 1
        c -= 1
        
    # 4. Bottom-Right: from (r_end + 1, c_end + 1) stepping (+1, +1)
    r, c = r_end + 1, c_end + 1
    while r < 4 * k and c < 4 * k and out[r, c] == 0:
        out[r, c] = 2
        r += 1
        c += 1
        
    return out

with open("training/469497ad.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_469497ad(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!\nPred:\n{pred}\nExpected:\n{expected}"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_469497ad(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!\nPred:\n{pred}\nExpected:\n{expected}"
print("469497ad: 100% PASS ON ALL TRAIN AND TEST!")
