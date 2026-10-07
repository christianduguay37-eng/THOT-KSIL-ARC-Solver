from thot_arc_core import ArcTask, ArcGrid
import numpy as np

task = ArcTask.load_from_file("Atelier_ARC_Prize/training/06df4c85.json")

def solve_connect_aligned_blocks(inp, ctx):
    h, w = inp.shape
    # Trouver la couleur de la grille (ex: 8 ou 1)
    grid_cols = [c for c in range(w) if len(np.unique(inp[:, c])) == 1 and inp[0, c] != 0]
    grid_rows = [r for r in range(h) if len(np.unique(inp[r, :])) == 1 and inp[r, 0] != 0]
    if not grid_cols or not grid_rows:
        return None
    grid_color = inp[grid_rows[0], 0]
    
    # Identifier la taille des cellules (dans ces exemples, 2x2)
    step_c = grid_cols[1] - grid_cols[0] if len(grid_cols) > 1 else 3
    step_r = grid_rows[1] - grid_rows[0] if len(grid_rows) > 1 else 3
    
    # Extraire les blocs 2x2 non nuls et différents de la grille
    out = inp.copy()
    
    # Détecter tous les blocs colorés
    objs = ArcGrid.get_connected_components(inp, background=0)
    blocks = [o for o in objs if o["color"] != grid_color and o["area"] == 4]
    
    # Grouper par couleur
    by_color = {}
    for b in blocks:
        by_color.setdefault(b["color"], []).append(b)
        
    for color, blist in by_color.items():
        if len(blist) < 2:
            continue
        # Vérifier paires alignées horizontalement (même min_r)
        for i in range(len(blist)):
            for j in range(i + 1, len(blist)):
                b1, b2 = blist[i], blist[j]
                r1, c1, h1, w1 = b1["bbox"]
                r2, c2, h2, w2 = b2["bbox"]
                
                # Même rangée
                if r1 == r2:
                    min_c = min(c1, c2)
                    max_c = max(c1, c2)
                    # Relier toutes les cellules de cette rangée entre min_c et max_c
                    for c in range(min_c, max_c + w1, step_c):
                        out[r1:r1+h1, c:c+w1] = color
                        
                # Même colonne
                elif c1 == c2:
                    min_r = min(r1, r2)
                    max_r = max(r1, r2)
                    for r in range(min_r, max_r + h1, step_r):
                        out[r:r+h1, c1:c1+w1] = color
                        
    return out

all_ok = True
for i, p in enumerate(task.train_pairs):
    pred = solve_connect_aligned_blocks(p["input"], task.train_pairs)
    match = pred is not None and np.array_equal(pred, p["output"])
    print(f"Train {i+1} : {match}")
    if not match: all_ok = False

pred_test = solve_connect_aligned_blocks(task.test_pairs[0]["input"], task.train_pairs)
match_test = pred_test is not None and np.array_equal(pred_test, task.test_pairs[0]["output"])
print(f"Test Pass : {match_test}")
