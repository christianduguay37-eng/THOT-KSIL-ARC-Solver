from thot_arc_core import ArcTask
import numpy as np

task = ArcTask.load_from_file("Atelier_ARC_Prize/training/0ca9ddb6.json")

def solve_satellite_cross(inp, ctx):
    h, w = inp.shape
    out = inp.copy()
    
    # Règle déduite :
    # Si pixel == 2 -> placer 4 aux 4 coins diagonaux
    # Si pixel == 1 -> placer 7 aux 4 points cardinaux
    for r in range(h):
        for c in range(w):
            val = inp[r, c]
            if val == 2:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = 4
            elif val == 1:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = 7
                        
    # Remettre les pixels originaux au-dessus
    for r in range(h):
        for c in range(w):
            if inp[r, c] != 0:
                out[r, c] = inp[r, c]
                
    return out

all_ok = True
for i, p in enumerate(task.train_pairs):
    pred = solve_satellite_cross(p["input"], task.train_pairs)
    match = pred is not None and np.array_equal(pred, p["output"])
    print(f"Train {i+1} : {match}")
    if not match: all_ok = False

pred_test = solve_satellite_cross(task.test_pairs[0]["input"], task.train_pairs)
match_test = pred_test is not None and np.array_equal(pred_test, task.test_pairs[0]["output"])
print(f"Test Pass : {match_test}")
