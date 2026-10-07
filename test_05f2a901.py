from thot_arc_core import ArcTask, ArcGrid
import numpy as np

task = ArcTask.load_from_file("Atelier_ARC_Prize/training/05f2a901.json")

def solve_dock_to_target(inp, ctx):
    objs = ArcGrid.get_connected_components(inp, background=0)
    if len(objs) != 2:
        return None
    # Un objet est la cible (couleur 8 par exemple), l'autre est le mobile (couleur 2)
    # Dans le train, la couleur 2 bouge et la 8 reste fixe
    mobile_obj = [o for o in objs if o["color"] == 2]
    target_obj = [o for o in objs if o["color"] != 2]
    if not mobile_obj or not target_obj:
        return None
    m = mobile_obj[0]
    t = target_obj[0]
    
    # Tester les 4 directions de glissement : Bas, Droite, Haut, Gauche
    h, w = inp.shape
    best_out = None
    
    for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        # Faire glisser m pas par pas jusqu'à toucher t ou le bord
        cur_coords = list(m["coords"])
        valid_shift = None
        for step in range(1, max(h, w)):
            next_coords = [(r + dr*step, c + dc*step) for r, c in m["coords"]]
            # Vérifier collision avec bords
            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in next_coords):
                break
            # Vérifier collision/contact avec t
            # Si next_coords intersecte t, stop avant
            if any(t["mask"][r, c] for r, c in next_coords):
                valid_shift = step - 1
                break
            # Vérifier si next_coords est adjacent à t (contact)
            adjacent = False
            for r, c in next_coords:
                for adj_r, adj_c in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                    if 0 <= adj_r < h and 0 <= adj_c < w and t["mask"][adj_r, adj_c]:
                        adjacent = True
                        break
                if adjacent: break
            if adjacent:
                valid_shift = step
                break
                
        if valid_shift is not None and valid_shift > 0:
            out = np.zeros_like(inp)
            # Dessiner t
            out[t["mask"]] = t["color"]
            # Dessiner m déplacé
            for r, c in m["coords"]:
                out[r + dr*valid_shift, c + dc*valid_shift] = m["color"]
            best_out = out
            break
            
    return best_out

all_ok = True
for i, p in enumerate(task.train_pairs):
    pred = solve_dock_to_target(p["input"], task.train_pairs)
    match = pred is not None and np.array_equal(pred, p["output"])
    print(f"Train {i+1} : {match}")
    if not match: all_ok = False

pred_test = solve_dock_to_target(task.test_pairs[0]["input"], task.train_pairs)
match_test = pred_test is not None and np.array_equal(pred_test, task.test_pairs[0]["output"])
print(f"Test Pass : {match_test}")
