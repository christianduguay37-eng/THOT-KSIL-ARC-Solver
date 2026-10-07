import json
import numpy as np

def solve_c8cbb738(inp):
    bg = int(np.argmax(np.bincount(inp.flatten())))
    
    # Analyze each non-bg color
    square_color = None
    square_size = None
    diamond_color = None
    h_rects = [] # (color, h) where width == S-1
    v_rects = [] # (color, w) where height == S-1
    
    colors = [c for c in np.unique(inp) if c != bg]
    
    for c in colors:
        pts = np.argwhere(inp == c)
        if len(pts) != 4:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        dr = r1 - r0
        dc = c1 - c0
        
        # Check if corners of rectangle
        corner_cells = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        actual_cells = {tuple(p) for p in pts}
        
        if actual_cells == corner_cells:
            if dr == dc:
                square_color = c
                square_size = dr + 1
            else:
                # rectangle
                pass
        else:
            # Diamond / cross: check if midpoints
            diamond_color = c

    # Second pass for rectangles now that square_size is known
    S = square_size
    mid = (S - 1) // 2
    
    for c in colors:
        if c == square_color or c == diamond_color:
            continue
        pts = np.argwhere(inp == c)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        dr = r1 - r0
        dc = c1 - c0
        if dr == S - 1:
            # Vertical rectangle spanning full height S-1, width dc
            v_rects.append((c, dc))
        elif dc == S - 1:
            # Horizontal rectangle spanning full width S-1, height dr
            h_rects.append((c, dr))
            
    out = np.full((S, S), bg, dtype=int)
    
    # 1. 4 corners
    out[0, 0] = square_color
    out[0, S - 1] = square_color
    out[S - 1, 0] = square_color
    out[S - 1, S - 1] = square_color
    
    # 2. Diamond / cross midpoints
    if diamond_color is not None:
        out[0, mid] = diamond_color
        out[S - 1, mid] = diamond_color
        out[mid, 0] = diamond_color
        out[mid, S - 1] = diamond_color
        
    # 3. Vertical rectangles (top & bottom edges at mid +- dc//2)
    for c, dc in v_rects:
        offset = dc // 2
        out[0, mid - offset] = c
        out[0, mid + offset] = c
        out[S - 1, mid - offset] = c
        out[S - 1, mid + offset] = c
        
    # 4. Horizontal rectangles (left & right edges at mid +- dr//2)
    for c, dr in h_rects:
        offset = dr // 2
        out[mid - offset, 0] = c
        out[mid + offset, 0] = c
        out[mid - offset, S - 1] = c
        out[mid + offset, S - 1] = c
        
    return out

with open("training/c8cbb738.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_c8cbb738(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!\nPred:\n{pred}\nExpected:\n{expected}"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_c8cbb738(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!\nPred:\n{pred}\nExpected:\n{expected}"
print("c8cbb738: 100% PASS ON ALL TRAIN AND TEST!")
