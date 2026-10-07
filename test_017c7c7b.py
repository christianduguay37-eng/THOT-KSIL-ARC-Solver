from thot_arc_core import ArcTask
import numpy as np

task = ArcTask.load_from_file("Atelier_ARC_Prize/training/017c7c7b.json")

def solve_periodic_rows(inp, ctx):
    h, w = inp.shape
    target_h = 9
    # Trouver la période P de répétition des lignes dans inp
    # Pour P in [1, 2, 3, 4, 5]
    best_P = None
    for P in range(1, h):
        is_periodic = True
        for r in range(P, h):
            if not np.array_equal(inp[r], inp[r % P]):
                is_periodic = False
                break
        if is_periodic:
            best_P = P
            break
            
    if best_P is None:
        return None
        
    out = np.zeros((target_h, w), dtype=np.int32)
    for r in range(target_h):
        out[r] = inp[r % best_P]
        
    # Recoloration 1 -> 2
    out[out == 1] = 2
    return out

all_ok = True
for i, p in enumerate(task.train_pairs):
    pred = solve_periodic_rows(p["input"], task.train_pairs)
    match = pred is not None and np.array_equal(pred, p["output"])
    print(f"Train {i+1} : {match}")
    if not match: all_ok = False

pred_test = solve_periodic_rows(task.test_pairs[0]["input"], task.train_pairs)
match_test = pred_test is not None and np.array_equal(pred_test, task.test_pairs[0]["output"])
print(f"Test Pass : {match_test}")
