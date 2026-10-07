import json
import numpy as np
from collections import deque

def solve_9edfc990(inp):
    inp = np.array(inp)
    H, W = inp.shape
    out = inp.copy()
    
    q = deque()
    for r in range(H):
        for c in range(W):
            if inp[r, c] == 1:
                q.append((r, c))
                
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and out[nr, nc] == 0:
                out[nr, nc] = 1
                q.append((nr, nc))
                
    return out

if __name__ == "__main__":
    with open("training/9edfc990.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_9edfc990(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_9edfc990(inp)
        print(f"Test {i} output shape: {pred.shape}")
        
    if all_ok:
        print("ALL TESTS PASSED for 9edfc990!")
