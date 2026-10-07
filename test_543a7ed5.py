import json
import numpy as np
from scipy.ndimage import binary_dilation

def solve_543a7ed5(inp):
    out = inp.copy()
    H, W = inp.shape
    # Background is 8, foreground is 6
    # Any 8 that is completely enclosed by 6 becomes 4
    # To find holes enclosed by 6:
    # An 8 is background/outside if it can reach the boundary without touching 6
    visited = np.zeros((H, W), dtype=bool)
    # Flood fill from boundaries through 8s
    from collections import deque
    q = deque()
    for r in range(H):
        for c in [0, W - 1]:
            if inp[r, c] == 8 and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))
    for c in range(W):
        for r in [0, H - 1]:
            if inp[r, c] == 8 and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))
                
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if inp[nr, nc] == 8 and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
                    
    # Holes are 8s that were NOT visited!
    holes = (inp == 8) & (~visited)
    out[holes] = 4
    
    # Now halo of 3s: 8-neighborhood of 6s, only on outside 8s (i.e. visited & inp == 8)
    # Structuring element 3x3 for 8-connectivity
    struct = np.ones((3, 3), dtype=bool)
    dilated_6 = binary_dilation(inp == 6, structure=struct)
    
    # Cells that are in dilated_6, and were visited 8s
    halo = dilated_6 & (inp == 8) & visited
    out[halo] = 3
    
    return out

with open("training/543a7ed5.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_543a7ed5(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_543a7ed5(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("543a7ed5: 100% PASS ON ALL TRAIN AND TEST!")
