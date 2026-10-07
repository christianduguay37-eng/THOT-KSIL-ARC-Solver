import json
import numpy as np
from scipy.ndimage import label

def solve_234bbc79(inp):
    if 5 not in inp:
        return None
    rec = inp.copy()
    for _ in range(5):
        for r, c in np.argwhere(rec == 5):
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1] and rec[nr, nc] not in (0, 5):
                    rec[r, c] = rec[nr, nc]
                    break

    non_zero_cols = [c for c in range(inp.shape[1]) if not np.all(inp[:, c] == 0)]
    compact = rec[:, non_zero_cols]
    lbl, num = label(inp != 0)
    pieces = []
    anchor_idx = None
    for i in range(1, num+1):
        pts_inp = np.argwhere(lbl == i)
        cols_in_piece = [inp[r, c] for r, c in pts_inp if inp[r, c] != 5]
        piece_color = cols_in_piece[0]
        pts_compact = []
        for r, c in pts_inp:
            new_c = non_zero_cols.index(c)
            pts_compact.append((r, new_c))
        pts_arr = np.array(pts_compact)
        if 0 in pts_arr[:, 1]:
            anchor_idx = len(pieces)
        pieces.append((piece_color, pts_arr))

    anchor_rows = set(pieces[anchor_idx][1][:, 0])
    has_height_3 = any((pts[:, 0].max() - pts[:, 0].min() + 1 == 3) for _, pts in pieces)
    if has_height_3:
        target_rows = {0, 1, 2}
    else:
        target_rows = anchor_rows

    valid_configs = []
    def search_placement(p_idx, current_grid, shifts):
        if p_idx == len(pieces):
            _, n_comp = label(current_grid != 0)
            if n_comp == 1:
                valid_configs.append((current_grid.copy(), list(shifts)))
            return
        col, pts = pieces[p_idx]
        min_r = pts[:, 0].min()
        max_r = pts[:, 0].max()
        for dr in range(-min_r, 3 - max_r):
            if p_idx == anchor_idx and dr != 0:
                continue
            shifted = pts + [dr, 0]
            if not set(shifted[:, 0]).issubset(target_rows):
                continue
            if np.all(current_grid[shifted[:, 0], shifted[:, 1]] == 0):
                current_grid[shifted[:, 0], shifted[:, 1]] = col
                search_placement(p_idx + 1, current_grid, shifts + [dr])
                current_grid[shifted[:, 0], shifted[:, 1]] = 0

    grid0 = np.zeros((3, len(non_zero_cols)), dtype=int)
    search_placement(0, grid0, [])

    def cm_score(entry):
        g, shifts = entry
        cm_r = np.mean(np.argwhere(g != 0)[:, 0])
        return abs(cm_r - 1.0)

    valid_configs.sort(key=cm_score)
    return valid_configs[0][0]

if __name__ == "__main__":
    with open("training/234bbc79.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        assert np.array_equal(solve_234bbc79(np.array(p["input"])), np.array(p["output"]))
    for i, p in enumerate(d["test"]):
        assert np.array_equal(solve_234bbc79(np.array(p["input"])), np.array(p["output"]))
    print("234bbc79: 100% PASS ON ALL TRAIN AND TEST!")
