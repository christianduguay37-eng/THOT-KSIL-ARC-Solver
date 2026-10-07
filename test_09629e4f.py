from thot_arc_core import ArcTask
import numpy as np

task = ArcTask.load_from_file("Atelier_ARC_Prize/training/09629e4f.json")

def solve_subgrid_minimap(inp, ctx):
    h, w = inp.shape
    # Grille 11x11 séparée par des 5 aux lignes 3 et 7, et colonnes 3 et 7
    # Les 9 quadrants 3x3 :
    # row ranges: (0..3), (4..7), (8..11)
    # col ranges: (0..3), (4..7), (8..11)
    r_slices = [(0, 3), (4, 7), (8, 11)]
    c_slices = [(0, 3), (4, 7), (8, 11)]
    
    # Trouver quel quadrant N'A PAS la couleur sentinelle 8
    key_quadrant = None
    for qr, (r1, r2) in enumerate(r_slices):
        for qc, (c1, c2) in enumerate(c_slices):
            sub = inp[r1:r2, c1:c2]
            if 8 not in sub:
                key_quadrant = sub
                break
        if key_quadrant is not None:
            break
            
    if key_quadrant is None:
        return None
        
    out = np.full((11, 11), fill_value=5, dtype=np.int32)
    for qr, (r1, r2) in enumerate(r_slices):
        for qc, (c1, c2) in enumerate(c_slices):
            fill_color = key_quadrant[qr, qc]
            out[r1:r2, c1:c2] = fill_color
            
    return out

all_ok = True
for i, p in enumerate(task.train_pairs):
    pred = solve_subgrid_minimap(p["input"], task.train_pairs)
    match = pred is not None and np.array_equal(pred, p["output"])
    print(f"Train {i+1} : {match}")
    if not match: all_ok = False

pred_test = solve_subgrid_minimap(task.test_pairs[0]["input"], task.train_pairs)
match_test = pred_test is not None and np.array_equal(pred_test, task.test_pairs[0]["output"])
print(f"Test Pass : {match_test}")
