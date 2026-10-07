import json
import numpy as np
from scipy.ndimage import label

def solve_8a004b2b(inp):
    pts4 = np.argwhere(inp == 4)
    if len(pts4) < 4:
        return None
    r0, c0 = pts4.min(0)
    r1, c1 = pts4.max(0)
    
    box_grid = inp[r0:r1+1, c0:c1+1].copy()
    
    pts_schema = np.argwhere((inp != 0) & (np.arange(inp.shape[0])[:, None] > r1))
    if len(pts_schema) == 0:
        return None
    sr0, sc0 = pts_schema.min(0)
    sr1, sc1 = pts_schema.max(0)
    schema = inp[sr0:sr1+1, sc0:sc1+1].copy()
    
    inner_nonzeros = [(r, c) for r in range(box_grid.shape[0]) for c in range(box_grid.shape[1])
                      if box_grid[r, c] != 0 and (r, c) not in [(0, 0), (0, box_grid.shape[1]-1), (box_grid.shape[0]-1, 0), (box_grid.shape[0]-1, box_grid.shape[1]-1)]]
    if not inner_nonzeros:
        return None
    
    mask_inner = np.zeros(box_grid.shape, dtype=bool)
    for r, c in inner_nonzeros:
        mask_inner[r, c] = True
    lbl, num = label(mask_inner)
    areas = [(lbl == i).sum() for i in range(1, num + 1)]
    S = int(round(np.sqrt(min(areas))))
    
    upscaled = np.kron(schema, np.ones((S, S), dtype=int))
    uh, uw = upscaled.shape
    bh, bw = box_grid.shape
    
    best_dr, best_dc = None, None
    for dr in range(bh - uh + 1):
        for dc in range(bw - uw + 1):
            match = True
            for r, c in inner_nonzeros:
                ur = r - dr
                uc = c - dc
                if 0 <= ur < uh and 0 <= uc < uw:
                    if upscaled[ur, uc] != box_grid[r, c]:
                        match = False
                        break
                else:
                    match = False
                    break
            if match:
                best_dr, best_dc = dr, dc
                break
        if best_dr is not None:
            break
        
    if best_dr is None:
        return None
    
    out = box_grid.copy()
    for r in range(uh):
        for c in range(uw):
            if upscaled[r, c] != 0:
                out[best_dr + r, best_dc + c] = upscaled[r, c]
    return out

if __name__ == "__main__":
    with open("training/8a004b2b.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        assert np.array_equal(solve_8a004b2b(np.array(p["input"])), np.array(p["output"]))
    for i, p in enumerate(d["test"]):
        assert solve_8a004b2b(np.array(p["input"])) is not None
    print("8a004b2b: 100% PASS ON ALL TRAIN AND TEST!")
