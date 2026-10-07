import json
import numpy as np
from scipy.ndimage import label

def solve_e6721834(grid):
    inp = np.array(grid)
    H, W = inp.shape
    candidates = []
    if H % 2 == 0:
        h = H // 2
        candidates.append(('H', inp[:h, :], inp[h:, :]))
    if W % 2 == 0:
        w = W // 2
        candidates.append(('V', inp[:, :w], inp[:, w:]))
        
    best_split = None
    for axis, h1, h2 in candidates:
        v1, c1 = np.unique(h1, return_counts=True)
        bg1 = v1[np.argmax(c1)]
        fg1 = set(v1) - {bg1}
        v2, c2 = np.unique(h2, return_counts=True)
        bg2 = v2[np.argmax(c2)]
        fg2 = set(v2) - {bg2}
        if fg1.issubset(fg2) and not fg2.issubset(fg1):
            best_split = (h1, bg1, fg1, h2, bg2, fg2)
            break
        elif fg2.issubset(fg1) and not fg1.issubset(fg2):
            best_split = (h2, bg2, fg2, h1, bg1, fg1)
            break
            
    target, target_bg, anchor_colors, shape_grid, shape_bg, _ = best_split
    th, tw = target.shape
    labeled, num = label(shape_grid != shape_bg)
    
    comp_options = []
    for c_id in range(1, num + 1):
        c_mask = (labeled == c_id)
        rows, cols = np.where(c_mask)
        shape_anchors = [(r, c, shape_grid[r, c]) for r, c in zip(rows, cols) if shape_grid[r, c] in anchor_colors]
        valid_shifts = []
        for dr in range(-th, th):
            for dc in range(-tw, tw):
                match = True
                matched_anchors = set()
                for ar, ac, acol in shape_anchors:
                    tr = ar + dr
                    tc = ac + dc
                    if 0 <= tr < th and 0 <= tc < tw and target[tr, tc] == acol:
                        matched_anchors.add((tr, tc))
                    else:
                        match = False
                        break
                if match and shape_anchors:
                    if all(0 <= r + dr < th and 0 <= c + dc < tw for r, c in zip(rows, cols)):
                        shape_pixels = {(r + dr, c + dc) for r, c in zip(rows, cols)}
                        valid_shifts.append((dr, dc, matched_anchors, shape_pixels))
        if valid_shifts:
            comp_options.append((c_id, valid_shifts))
            
    used_anchors = set()
    used_pixels = set()
    chosen_placements = []
    
    comp_options.sort(key=lambda item: max(len(s[2]) for s in item[1]), reverse=True)
    
    for c_id, shifts in comp_options:
        best_shift = None
        for dr, dc, m_anchors, s_pixels in sorted(shifts, key=lambda s: len(s[2]), reverse=True):
            if not (m_anchors & used_anchors) and not (s_pixels & used_pixels):
                best_shift = (dr, dc, m_anchors, s_pixels, c_id)
                break
        if best_shift:
            dr, dc, m_anchors, s_pixels, c_id = best_shift
            used_anchors |= m_anchors
            used_pixels |= s_pixels
            chosen_placements.append((c_id, dr, dc))
            
    pred = target.copy()
    for c_id, dr, dc in chosen_placements:
        c_mask = (labeled == c_id)
        rows, cols = np.where(c_mask)
        for r, c in zip(rows, cols):
            pred[r + dr, c + dc] = shape_grid[r, c]
            
    return pred.tolist()

if __name__ == "__main__":
    with open("training/e6721834.json") as f:
        task = json.load(f)
    for idx, ex in enumerate(task["train"]):
        res = solve_e6721834(ex["input"])
        assert res == ex["output"], f"Train {idx} failed!"
        print(f"Train {idx} PASS!")
    for idx, ex in enumerate(task["test"]):
        res = solve_e6721834(ex["input"])
        assert res == ex["output"], f"Test {idx} failed!"
        print(f"Test {idx} output shape: {len(res)}x{len(res[0])}")
        print("Test 0 SUCCESS!")
