import json
import numpy as np

def solve_928ad970(inp):
    out = inp.copy()
    H, W = inp.shape
    
    # Find the positions of the 4 gray dots (5s)
    fives = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 5]
    assert len(fives) == 4, f"Expected 4 fives, got {len(fives)}"
    
    # Sort fives
    # Top five has min row
    top_r = min(r for r, c in fives)
    bot_r = max(r for r, c in fives)
    left_c = min(c for r, c in fives)
    right_c = max(c for r, c in fives)
    
    # The box color is the other nonzero color
    colors = [c for c in np.unique(inp) if c != 0 and c != 5]
    assert len(colors) == 1
    C = colors[0]
    
    box_r0 = top_r + 1
    box_r1 = bot_r - 1
    box_c0 = left_c + 1
    box_c1 = right_c - 1
    
    # Draw hollow box of color C
    out[box_r0, box_c0:box_c1+1] = C
    out[box_r1, box_c0:box_c1+1] = C
    out[box_r0:box_r1+1, box_c0] = C
    out[box_r0:box_r1+1, box_c1] = C
    
    # Keep the original 5s intact (in case the box overlapped, but here box is offset by 1)
    for r, c in fives:
        out[r, c] = 5
        
    return out

with open("training/928ad970.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_928ad970(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_928ad970(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("928ad970: 100% PASS ON ALL TRAIN AND TEST!")
