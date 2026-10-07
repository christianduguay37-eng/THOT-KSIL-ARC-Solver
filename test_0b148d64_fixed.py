import sys
from thot_arc_core import ArcTask
import numpy as np

task = ArcTask.load_from_file("Atelier_ARC_Prize/training/0b148d64.json")

def solve_odd_subgrid_by_separators(inp, ctx):
    h, w = inp.shape
    # Trouver les lignes complètes de 0
    zero_rows = [r for r in range(h) if np.all(inp[r, :] == 0)]
    zero_cols = [c for c in range(w) if np.all(inp[:, c] == 0)]
    
    if not zero_rows and not zero_cols:
        return None
        
    # Segmenter les rangées
    row_blocks = []
    start_r = 0
    for r in range(h):
        if r in zero_rows:
            if r > start_r:
                row_blocks.append((start_r, r))
            start_r = r + 1
    if start_r < h:
        row_blocks.append((start_r, h))
        
    # Segmenter les colonnes
    col_blocks = []
    start_c = 0
    for c in range(w):
        if c in zero_cols:
            if c > start_c:
                col_blocks.append((start_c, c))
            start_c = c + 1
    if start_c < w:
        col_blocks.append((start_c, w))
        
    # Extraire tous les sous-blocs
    blocks = []
    for r1, r2 in row_blocks:
        for c1, c2 in col_blocks:
            sub = inp[r1:r2, c1:c2]
            colors = set(sub[sub != 0])
            if colors:
                blocks.append({"sub": sub, "colors": colors})
                
    if not blocks:
        return None
        
    # Compter les occurrences de chaque couleur à travers les blocs
    color_block_counts = {}
    for b in blocks:
        for c in b["colors"]:
            color_block_counts[c] = color_block_counts.get(c, 0) + 1
            
    # Trouver le bloc qui a une couleur unique
    unique_color_blocks = [b for b in blocks if any(color_block_counts[c] == 1 for c in b["colors"])]
    if len(unique_color_blocks) == 1:
        return unique_color_blocks[0]["sub"]
        
    return None

all_ok = True
for i, p in enumerate(task.train_pairs):
    pred = solve_odd_subgrid_by_separators(p["input"], task.train_pairs)
    match = pred is not None and np.array_equal(pred, p["output"])
    print(f"Train {i+1} : {match}")
    if not match: all_ok = False

pred_test = solve_odd_subgrid_by_separators(task.test_pairs[0]["input"], task.train_pairs)
match_test = pred_test is not None and np.array_equal(pred_test, task.test_pairs[0]["output"])
print(f"Test Pass : {match_test}")
