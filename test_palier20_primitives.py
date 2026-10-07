import sys
import os
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from thot_arc_core import ArcTask, ArcGrid

def test_odd_subgrid():
    task = ArcTask.load_from_file("Atelier_ARC_Prize/training/0b148d64.json")
    def solve(inp, ctx):
        h, w = inp.shape
        # Découper par les lignes et colonnes de zéros complètes
        non_zero_rows = np.where(np.any(inp != 0, axis=1))[0]
        non_zero_cols = np.where(np.any(inp != 0, axis=0))[0]
        # Trouver les composantes connexes de blocs
        objs = ArcGrid.get_connected_components(inp, background=0)
        # Chaque objet a une couleur dominante
        colors = [o["color"] for o in objs]
        # Trouver la couleur unique (minoritaire)
        unique_colors = [c for c in set(colors) if colors.count(c) == 1]
        if unique_colors:
            odd_color = unique_colors[0]
            odd_obj = [o for o in objs if o["color"] == odd_color][0]
            return odd_obj["sub_grid"]
        return None

    all_ok = True
    for p in task.train_pairs:
        pred = solve(p["input"], task.train_pairs)
        if pred is None or not np.array_equal(pred, p["output"]):
            all_ok = False
            break
    print(f"Test 0b148d64 (Volet discordant) : {'REUSSI' if all_ok else 'ECHEC'}")
    pred_test = solve(task.test_pairs[0]["input"], task.train_pairs)
    match_test = np.array_equal(pred_test, task.test_pairs[0]["output"])
    print(f"Test 0b148d64 (Exact match test) : {match_test}")

def test_seed_stripes():
    task = ArcTask.load_from_file("Atelier_ARC_Prize/training/0a938d79.json")
    def solve(inp, ctx):
        h, w = inp.shape
        # Trouver les deux points graines non nuls
        coords = np.argwhere(inp != 0)
        if len(coords) != 2:
            return None
        # Trier par colonne
        coords = sorted(coords, key=lambda pt: pt[1])
        r1, c1 = coords[0]
        r2, c2 = coords[1]
        col1 = inp[r1, c1]
        col2 = inp[r2, c2]
        delta_c = c2 - c1
        if delta_c <= 0:
            return None
        out = np.zeros_like(inp)
        # Propager vers la droite
        cur_c = c1
        cur_col = col1
        alt = False
        while cur_c < w:
            out[:, cur_c] = col1 if not alt else col2
            alt = not alt
            cur_c += delta_c
        return out

    all_ok = True
    for p in task.train_pairs:
        pred = solve(p["input"], task.train_pairs)
        if pred is None or not np.array_equal(pred, p["output"]):
            all_ok = False
            break
    print(f"Test 0a938d79 (Rayures alternées par graine) : {'REUSSI' if all_ok else 'ECHEC'}")
    pred_test = solve(task.test_pairs[0]["input"], task.train_pairs)
    match_test = np.array_equal(pred_test, task.test_pairs[0]["output"])
    print(f"Test 0a938d79 (Exact match test) : {match_test}")

test_odd_subgrid()
test_seed_stripes()
