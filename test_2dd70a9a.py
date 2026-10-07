import json
import numpy as np

def solve_2dd70a9a(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    pts3 = list(zip(*np.where(grid == 3)))
    pts2 = list(zip(*np.where(grid == 2)))
    if len(pts3) != 2 or len(pts2) != 2:
        return grid
        
    r0, c0 = pts3[0]
    r1, c1 = pts3[1]
    dr = r1 - r0
    dc = c1 - c0
    candidates = []
    if dr != 0:
        candidates.append((min(r0, r1), c0, -1, 0))
        candidates.append((max(r0, r1), c0, 1, 0))
    else:
        candidates.append((r0, min(c0, c1), 0, -1))
        candidates.append((r0, max(c0, c1), 0, 1))
        
    set2 = set(pts2)
    r2_0, c2_0 = pts2[0]
    r2_1, c2_1 = pts2[1]
    d2c = c2_1 - c2_0
    
    valid_paths = []
    for sr, sc, init_dr, init_dc in candidates:
        curr_r, curr_c = sr, sc
        leg1 = []
        hit_8 = False
        while 0 <= curr_r + init_dr < H and 0 <= curr_c + init_dc < W:
            if grid[curr_r + init_dr, curr_c + init_dc] == 8:
                hit_8 = True
                break
            curr_r += init_dr
            curr_c += init_dc
            leg1.append((curr_r, curr_c))
        
        # Only turn if hit_8 is True!
        if not hit_8 or not leg1:
            continue
            
        turn_r, turn_c = curr_r, curr_c
        if d2c == 0:
            target_c = c2_0
            perp_dc = 1 if target_c > turn_c else (-1 if target_c < turn_c else 0)
            if perp_dc == 0:
                continue
            leg2 = []
            hit_obs = False
            c = turn_c
            while c != target_c:
                c += perp_dc
                if grid[turn_r, c] == 8:
                    hit_obs = True
                    break
                leg2.append((turn_r, c))
            if hit_obs:
                continue
            r_targets = [r for r, _ in pts2]
            target_r = min(r_targets, key=lambda r: abs(r - turn_r))
            dock_dr = 1 if target_r > turn_r else (-1 if target_r < turn_r else 0)
            if dock_dr == 0:
                continue
            leg3 = []
            r = turn_r
            docked = False
            while 0 <= r + dock_dr < H:
                r += dock_dr
                if (r, target_c) in set2:
                    docked = True
                    break
                if grid[r, target_c] == 8:
                    break
                leg3.append((r, target_c))
            if docked:
                valid_paths.append(leg1 + leg2 + leg3)
        else:
            target_r = r2_0
            perp_dr = 1 if target_r > turn_r else (-1 if target_r < turn_r else 0)
            if perp_dr == 0:
                continue
            leg2 = []
            hit_obs = False
            r = turn_r
            while r != target_r:
                r += perp_dr
                if grid[r, turn_c] == 8:
                    hit_obs = True
                    break
                leg2.append((r, turn_c))
            if hit_obs:
                continue
            c_targets = [c for _, c in pts2]
            target_c = min(c_targets, key=lambda c: abs(c - turn_c))
            dock_dc = 1 if target_c > turn_c else (-1 if target_c < turn_c else 0)
            if dock_dc == 0:
                continue
            leg3 = []
            c = turn_c
            docked = False
            while 0 <= c + dock_dc < W:
                c += dock_dc
                if (target_r, c) in set2:
                    docked = True
                    break
                if grid[target_r, c] == 8:
                    break
                leg3.append((target_r, c))
            if docked:
                valid_paths.append(leg1 + leg2 + leg3)
            
    out = grid.copy()
    if valid_paths:
        for pr, pc in valid_paths[0]:
            out[pr, pc] = 3
    return out

if __name__ == "__main__":
    with open("training/2dd70a9a.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_2dd70a9a(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_2dd70a9a(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 2dd70a9a!")
