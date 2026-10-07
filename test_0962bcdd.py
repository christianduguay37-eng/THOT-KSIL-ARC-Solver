from thot_arc_core import ArcTask
import numpy as np

task = ArcTask.load_from_file("Atelier_ARC_Prize/training/0962bcdd.json")

def solve_compass_star(inp, ctx):
    h, w = inp.shape
    out = inp.copy()
    
    # Trouver les centres des croix
    # Un centre est un pixel non nul dont les 4 voisins cardinaux sont d'une AUTRE couleur non nulle
    centers = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            val = inp[r, c]
            if val != 0:
                cardinals = [inp[r-1, c], inp[r+1, c], inp[r, c-1], inp[r, c+1]]
                if all(cv != 0 and cv != val for cv in cardinals) and len(set(cardinals)) == 1:
                    centers.append((r, c, val, cardinals[0]))
                    
    if not centers:
        return None
        
    for r, c, c_center, c_arm in centers:
        # Étendre les bras cardinaux à rayon 2
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in [1, 2]:
                nr, nc = r + dr * dist, c + dc * dist
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = c_arm
                    
        # Étendre les rayons diagonaux à rayon 2
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            for dist in [1, 2]:
                nr, nc = r + dr * dist, c + dc * dist
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = c_center
                    
        # Remettre le centre
        out[r, c] = c_center
        
    return out

all_ok = True
for i, p in enumerate(task.train_pairs):
    pred = solve_compass_star(p["input"], task.train_pairs)
    match = pred is not None and np.array_equal(pred, p["output"])
    print(f"Train {i+1} : {match}")
    if not match: all_ok = False

pred_test = solve_compass_star(task.test_pairs[0]["input"], task.train_pairs)
match_test = pred_test is not None and np.array_equal(pred_test, task.test_pairs[0]["output"])
print(f"Test Pass : {match_test}")
