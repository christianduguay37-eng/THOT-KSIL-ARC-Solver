from thot_arc_core import ArcTask
import numpy as np

task = ArcTask.load_from_file("Atelier_ARC_Prize/training/1190e5a7.json")

def solve_grid_cell_count(inp, ctx):
    h, w = inp.shape
    # Trouver la couleur de séparateur (la ligne complète)
    sep_colors = []
    h_seps = []
    v_seps = []
    for r in range(h):
        if len(np.unique(inp[r, :])) == 1:
            sep_colors.append(inp[r, 0])
            h_seps.append(r)
    for c in range(w):
        if len(np.unique(inp[:, c])) == 1:
            sep_colors.append(inp[0, c])
            v_seps.append(c)
            
    if not h_seps or not v_seps:
        return None
        
    sep_color = max(set(sep_colors), key=sep_colors.count)
    # Filtrer les lignes qui ont exactement la sep_color
    h_lines = [r for r in h_seps if inp[r, 0] == sep_color]
    v_lines = [c for c in v_seps if inp[0, c] == sep_color]
    
    num_rows = len(h_lines) + 1
    num_cols = len(v_lines) + 1
    
    # Trouver la couleur du contenu (différente de sep_color)
    content_colors = [c for c in np.unique(inp) if c != sep_color]
    if not content_colors:
        return None
    content_color = content_colors[0]
    
    out = np.full((num_rows, num_cols), fill_value=content_color, dtype=np.int32)
    return out

all_ok = True
for i, p in enumerate(task.train_pairs):
    pred = solve_grid_cell_count(p["input"], task.train_pairs)
    match = pred is not None and np.array_equal(pred, p["output"])
    print(f"Train {i+1} : {match}")
    if not match: all_ok = False

pred_test = solve_grid_cell_count(task.test_pairs[0]["input"], task.train_pairs)
match_test = pred_test is not None and np.array_equal(pred_test, task.test_pairs[0]["output"])
print(f"Test Pass : {match_test}")
