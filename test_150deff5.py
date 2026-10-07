import json
import numpy as np
from itertools import combinations

def solve_150deff5(inp):
    if 5 not in inp:
        return None
    H, W = inp.shape
    candidates = [(r, c) for r in range(H-1) for c in range(W-1) if np.all(inp[r:r+2, c:c+2] == 5)]
    if len(candidates) == 0 or len(candidates) > 10:
        return None
    
    def is_disjoint(sub):
        used = set()
        for r, c in sub:
            for dr in (0, 1):
                for dc in (0, 1):
                    if (r+dr, c+dc) in used: return False
                    used.add((r+dr, c+dc))
        return True

    def leaves_no_2x2(sub):
        mask = (inp == 5).copy()
        for r, c in sub:
            mask[r:r+2, c:c+2] = False
        for r in range(H-1):
            for c in range(W-1):
                if np.all(mask[r:r+2, c:c+2]): return False
        return True

    def can_partition_into_bars(rem_mask):
        coords = set(map(tuple, np.argwhere(rem_mask)))
        if len(coords) % 3 != 0: return False
        
        def backtrack(curr_coords):
            if not curr_coords: return True
            r, c = min(curr_coords)
            h_bar = {(r, c), (r, c+1), (r, c+2)}
            if h_bar.issubset(curr_coords):
                if backtrack(curr_coords - h_bar): return True
            v_bar = {(r, c), (r+1, c), (r+2, c)}
            if v_bar.issubset(curr_coords):
                if backtrack(curr_coords - v_bar): return True
            return False
            
        return backtrack(coords)

    # Search combinations of 2x2 blocks
    best_sub = None
    for k in range(1, len(candidates)+1):
        for sub in combinations(candidates, k):
            if is_disjoint(sub) and leaves_no_2x2(sub):
                rem_mask = (inp == 5).copy()
                for r, c in sub:
                    rem_mask[r:r+2, c:c+2] = False
                if can_partition_into_bars(rem_mask):
                    best_sub = sub
                    break
        if best_sub is not None:
            break

    out = np.zeros_like(inp)
    out[inp == 5] = 2
    if best_sub is not None:
        for r, c in best_sub:
            out[r:r+2, c:c+2] = 8
    return out

if __name__ == "__main__":
    with open("training/150deff5.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        assert np.array_equal(solve_150deff5(np.array(p["input"])), np.array(p["output"]))
    for i, p in enumerate(d["test"]):
        assert np.array_equal(solve_150deff5(np.array(p["input"])), np.array(p["output"]))
    print("150deff5: 100% PASS ON ALL TRAIN AND TEST!")
