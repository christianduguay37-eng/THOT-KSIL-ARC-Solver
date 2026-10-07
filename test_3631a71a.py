import json
import numpy as np

def solve_3631a71a(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    if 9 not in grid:
        return grid
    
    valid_ops = []
    
    # 1. transpose
    if H == W:
        conflicts = 0
        agreements = 0
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 9 and grid[c, r] != 9:
                    if grid[r, c] == grid[c, r]:
                        agreements += 1
                    else:
                        conflicts += 1
        if conflicts == 0 and agreements > 0:
            valid_ops.append(lambda r, c: (c, r))
            
    # 2. vertical reflections c -> 2*c0 - c
    for two_c0 in range(2 * W):
        conflicts = 0
        agreements = 0
        for r in range(H):
            for c in range(W):
                c_sym = two_c0 - c
                if 0 <= c_sym < W:
                    if grid[r, c] != 9 and grid[r, c_sym] != 9:
                        if grid[r, c] == grid[r, c_sym]:
                            agreements += 1
                        else:
                            conflicts += 1
        if conflicts == 0 and agreements > 0:
            valid_ops.append((lambda tc: lambda r, c: (r, tc - c))(two_c0))
            
    # 3. horizontal reflections r -> 2*r0 - r
    for two_r0 in range(2 * H):
        conflicts = 0
        agreements = 0
        for r in range(H):
            for c in range(W):
                r_sym = two_r0 - r
                if 0 <= r_sym < H:
                    if grid[r, c] != 9 and grid[r_sym, c] != 9:
                        if grid[r, c] == grid[r_sym, c]:
                            agreements += 1
                        else:
                            conflicts += 1
        if conflicts == 0 and agreements > 0:
            valid_ops.append((lambda tr: lambda r, c: (tr - r, c))(two_r0))
            
    # Iterative flood/propagation of known colors into 9
    rec = grid.copy()
    changed = True
    iterations = 0
    while changed and iterations < 100:
        changed = False
        iterations += 1
        for r in range(H):
            for c in range(W):
                if rec[r, c] == 9:
                    for op in valid_ops:
                        pr, pc = op(r, c)
                        if 0 <= pr < H and 0 <= pc < W and rec[pr, pc] != 9:
                            rec[r, c] = rec[pr, pc]
                            changed = True
                            break
    return rec

if __name__ == "__main__":
    with open("training/3631a71a.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_3631a71a(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_3631a71a(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 3631a71a!")
