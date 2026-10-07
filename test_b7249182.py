import json
import numpy as np

def solve_b7249182(inp):
    H, W = inp.shape
    out = np.zeros_like(inp)
    
    # Find the two dots
    dots = [(r, c, inp[r, c]) for r in range(H) for c in range(W) if inp[r, c] != 0]
    assert len(dots) == 2, f"Expected 2 dots, got {len(dots)}"
    
    (r1, c1, col1), (r2, c2, col2) = dots
    
    # Check if they are in the same column (vertical alignment)
    if c1 == c2:
        C = c1
        if r1 > r2:
            r1, r2 = r2, r1
            col1, col2 = col2, col1
            
        r_mid = (r1 + r2) // 2
        
        # Top shape (col1)
        # Stem
        out[r1:r_mid, C] = col1
        # Crossbar at r_mid - 1
        out[r_mid - 1, C - 2:C + 3] = col1
        # Fingers pointing down at r_mid
        out[r_mid, C - 2] = col1
        out[r_mid, C + 2] = col1
        
        # Bottom shape (col2)
        # Stem
        out[r_mid + 2:r2 + 1, C] = col2
        # Crossbar at r_mid + 2
        out[r_mid + 2, C - 2:C + 3] = col2
        # Fingers pointing up at r_mid + 1
        out[r_mid + 1, C - 2] = col2
        out[r_mid + 1, C + 2] = col2
        
    elif r1 == r2:
        R = r1
        if c1 > c2:
            c1, c2 = c2, c1
            col1, col2 = col2, col1
            
        c_mid = (c1 + c2) // 2
        
        # Left shape (col1)
        # Stem
        out[R, c1:c_mid] = col1
        # Crossbar at c_mid - 1
        out[R - 2:R + 3, c_mid - 1] = col1
        # Fingers pointing right at c_mid
        out[R - 2, c_mid] = col1
        out[R + 2, c_mid] = col1
        
        # Right shape (col2)
        # Stem
        out[R, c_mid + 2:c2 + 1] = col2
        # Crossbar at c_mid + 2
        out[R - 2:R + 3, c_mid + 2] = col2
        # Fingers pointing left at c_mid + 1
        out[R - 2, c_mid + 1] = col2
        out[R + 2, c_mid + 1] = col2
        
    return out

with open("training/b7249182.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_b7249182(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_b7249182(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("b7249182: 100% PASS ON ALL TRAIN AND TEST!")
