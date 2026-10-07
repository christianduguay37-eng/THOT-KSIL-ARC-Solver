from thot_arc_core import ArcTask
import numpy as np

task = ArcTask.load_from_file("Atelier_ARC_Prize/training/0dfd9992.json")

def solve_periodic_inpainting_2d(inp, ctx):
    h, w = inp.shape
    # Chercher la période (Pr, Pc) qui explique 100% des pixels non nuls
    best_pr, best_pc = None, None
    best_tile = None
    
    for Pr in range(2, min(12, h)):
        for Pc in range(2, min(12, w)):
            tile = np.zeros((Pr, Pc), dtype=np.int32)
            valid = True
            for r in range(h):
                for c in range(w):
                    val = inp[r, c]
                    if val != 0:
                        tr, tc = r % Pr, c % Pc
                        if tile[tr, tc] != 0 and tile[tr, tc] != val:
                            valid = False
                            break
                        tile[tr, tc] = val
                if not valid:
                    break
            # Vérifier si toutes les cases de la tuile sont remplies
            if valid and np.all(tile != 0):
                best_pr, best_pc = Pr, Pc
                best_tile = tile
                break
        if best_pr is not None:
            break
            
    if best_tile is None:
        return None
        
    out = inp.copy()
    for r in range(h):
        for c in range(w):
            if out[r, c] == 0:
                out[r, c] = best_tile[r % best_pr, c % best_pc]
    return out

all_ok = True
for i, p in enumerate(task.train_pairs):
    pred = solve_periodic_inpainting_2d(p["input"], task.train_pairs)
    match = pred is not None and np.array_equal(pred, p["output"])
    print(f"Train {i+1} : {match}")
    if not match: all_ok = False

pred_test = solve_periodic_inpainting_2d(task.test_pairs[0]["input"], task.train_pairs)
match_test = pred_test is not None and np.array_equal(pred_test, task.test_pairs[0]["output"])
print(f"Test Pass : {match_test}")
