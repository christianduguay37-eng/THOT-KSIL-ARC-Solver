import sys
from thot_arc_core import ArcTask
import numpy as np

task = ArcTask.load_from_file("Atelier_ARC_Prize/training/0a938d79.json")

def solve_seed_stripes(inp, ctx):
    coords = np.argwhere(inp != 0)
    if len(coords) != 2:
        return None
    pt1, pt2 = coords[0], coords[1]
    col1, col2 = inp[pt1[0], pt1[1]], inp[pt2[0], pt2[1]]
    h, w = inp.shape
    
    # Vérifier si c'est horizontal ou vertical
    # Si les deux points ont des colonnes différentes et des rangées aux extrémités
    dr = abs(pt2[0] - pt1[0])
    dc = abs(pt2[1] - pt1[1])
    
    out = np.zeros_like(inp)
    
    if dc > 0 and (pt1[0] in [0, h-1] and pt2[0] in [0, h-1]):
        # Bandes verticales
        pts = sorted([pt1, pt2], key=lambda p: p[1])
        c1, c2 = pts[0][1], pts[1][1]
        color1, color2 = inp[pts[0][0], c1], inp[pts[1][0], c2]
        delta = c2 - c1
        cur_c = c1
        alt = False
        while cur_c < w:
            out[:, cur_c] = color1 if not alt else color2
            alt = not alt
            cur_c += delta
        return out
        
    elif dr > 0 and (pt1[1] in [0, w-1] and pt2[1] in [0, w-1]):
        # Bandes horizontales
        pts = sorted([pt1, pt2], key=lambda p: p[0])
        r1, r2 = pts[0][0], pts[1][0]
        color1, color2 = inp[r1, pts[0][1]], inp[r2, pts[1][1]]
        delta = r2 - r1
        cur_r = r1
        alt = False
        while cur_r < h:
            out[cur_r, :] = color1 if not alt else color2
            alt = not alt
            cur_r += delta
        return out
        
    return None

all_ok = True
for i, p in enumerate(task.train_pairs):
    pred = solve_seed_stripes(p["input"], task.train_pairs)
    match = pred is not None and np.array_equal(pred, p["output"])
    print(f"Train {i+1} : {match}")
    if not match: all_ok = False

pred_test = solve_seed_stripes(task.test_pairs[0]["input"], task.train_pairs)
match_test = pred_test is not None and np.array_equal(pred_test, task.test_pairs[0]["output"])
print(f"Test Pass : {match_test}")
