import json
import numpy as np

def solve_484b58aa(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    
    shifts = []
    for dr in range(-15, 16):
        for dc in range(-15, 16):
            if dr == 0 and dc == 0:
                continue
            conflicts = 0
            agreements = 0
            for r in range(H):
                for c in range(W):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        v1 = grid[r, c]
                        v2 = grid[nr, nc]
                        if v1 != 0 and v2 != 0:
                            if v1 == v2:
                                agreements += 1
                            else:
                                conflicts += 1
                                break
                if conflicts > 0:
                    break
            if conflicts == 0 and agreements >= 20:
                shifts.append((dr, dc, agreements))
                
    shifts.sort(key=lambda s: s[2], reverse=True)
    shift_vecs = [(dr, dc) for dr, dc, _ in shifts]
    
    out = grid.copy()
    changed = True
    iterations = 0
    while changed and iterations < 100:
        changed = False
        iterations += 1
        for r in range(H):
            for c in range(W):
                if out[r, c] == 0:
                    for dr, dc in shift_vecs:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and out[nr, nc] != 0:
                            out[r, c] = out[nr, nc]
                            changed = True
                            break
    return out

if __name__ == "__main__":
    with open("training/484b58aa.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_484b58aa(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_484b58aa(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 484b58aa!")
