import json
import numpy as np
from itertools import product, combinations

def solve_0e206a2e(inp):
    non_zeros = [c for c in np.unique(inp) if c != 0]
    if len(non_zeros) < 4:
        return None
    counts = {c: (inp == c).sum() for c in non_zeros}
    carrier = max(counts, key=counts.get)
    anchor_colors = [c for c in non_zeros if c != carrier]
    if len(anchor_colors) != 3:
        return None
    pts_lens = [int(counts[c]) for c in anchor_colors]
    if len(set(pts_lens)) != 1 or pts_lens[0] not in (2, 4):
        return None
    pts_by_col = {c: [tuple(pt) for pt in np.argwhere(inp == c)] for c in anchor_colors}
    N = len(pts_by_col[anchor_colors[0]])
    
    all_tuples = list(product(*[pts_by_col[c] for c in anchor_colors]))
    compact_consts = []
    for tup in all_tuples:
        const = {c: np.array(pt) for c, pt in zip(anchor_colors, tup)}
        pts = list(const.values())
        max_d = max(abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) for p1 in pts for p2 in pts)
        if max_d <= 12:
            compact_consts.append(const)
            
    def get_transform(c1, c2):
        s0 = c1[anchor_colors[0]]
        t0 = c2[anchor_colors[0]]
        for flip in [1, -1]:
            for rot in range(4):
                cos_val = int(round(np.cos(rot * np.pi / 2)))
                sin_val = int(round(np.sin(rot * np.pi / 2)))
                R = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
                F = np.array([[1, 0], [0, flip]])
                M = R @ F
                if all(np.array_equal(M @ (c1[c] - s0), c2[c] - t0) for c in anchor_colors):
                    return lambda p: tuple(int(x) for x in (M @ (np.array(p) - s0) + t0))
        return None

    valid_partitions = []
    for subset in combinations(compact_consts, N):
        ok = True
        for c in anchor_colors:
            pts_in_sub = [tuple(sub[c]) for sub in subset]
            if len(set(pts_in_sub)) != N or set(pts_in_sub) != set(pts_by_col[c]):
                ok = False
                break
        if ok:
            valid_partitions.append(subset)
            
    carrier_pts = set(map(tuple, np.argwhere(inp == carrier)))
    best_pred = None
    for partition in valid_partitions:
        sources = []
        targets = []
        for const in partition:
            anchors = list(const.values())
            attached = {cp for cp in carrier_pts if min(abs(cp[0]-a[0]) + abs(cp[1]-a[1]) for a in anchors) <= 3}
            if attached: sources.append((const, attached))
            else: targets.append(const)
        if len(sources) == len(targets):
            grid = inp.copy()
            for const, att in sources:
                for c, pt in const.items():
                    grid[pt[0], pt[1]] = 0
                for r, c in att:
                    grid[r, c] = 0
            all_mapped = True
            for tgt in targets:
                matched = False
                for src, att in sources:
                    T = get_transform(src, tgt)
                    if T is not None:
                        for p in att:
                            tp = T(p)
                            grid[tp[0], tp[1]] = carrier
                        matched = True
                        break
                if not matched:
                    all_mapped = False
                    break
            if all_mapped:
                best_pred = grid
                break
                
    return best_pred

if __name__ == "__main__":
    with open("training/0e206a2e.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        assert np.array_equal(solve_0e206a2e(np.array(p["input"])), np.array(p["output"]))
    for i, p in enumerate(d["test"]):
        assert np.array_equal(solve_0e206a2e(np.array(p["input"])), np.array(p["output"]))
    print("0e206a2e: 100% PASS ON ALL TRAIN AND TEST!")
