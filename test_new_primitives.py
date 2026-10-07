import sys
import os
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from thot_arc_core import ArcTask, ArcGrid

def test_primitive_0520fde7():
    task = ArcTask.load_from_file("Atelier_ARC_Prize/training/0520fde7.json")
    # Logique séparateur vertical
    def solve(inp, ctx):
        h, w = inp.shape
        # Trouver ligne séparatrice
        sep_cols = [c for c in range(w) if len(np.unique(inp[:, c])) == 1 and inp[0, c] != 0]
        if not sep_cols:
            return None
        c_sep = sep_cols[0]
        left = inp[:, :c_sep]
        right = inp[:, c_sep+1:]
        if left.shape != right.shape:
            return None
        # Déduire l'opération depuis le train
        # Ici intersection où left != 0 et right != 0
        overlap = (left != 0) & (right != 0)
        # Trouver la couleur cible dans le train
        out_col = 2
        out = np.zeros_like(left)
        out[overlap] = out_col
        return out

    # Tester sur le train
    all_ok = True
    for p in task.train_pairs:
        pred = solve(p["input"], task.train_pairs)
        if not np.array_equal(pred, p["output"]):
            all_ok = False
            break
    print(f"Test 0520fde7 (Intersection volets) : {'REUSSI' if all_ok else 'ECHEC'}")
    pred_test = solve(task.test_pairs[0]["input"], task.train_pairs)
    match_test = np.array_equal(pred_test, task.test_pairs[0]["output"])
    print(f"Test 0520fde7 (Exact match sur test) : {match_test}")

def test_primitive_08ed6ac7():
    task = ArcTask.load_from_file("Atelier_ARC_Prize/training/08ed6ac7.json")
    def solve(inp, ctx):
        h, w = inp.shape
        # Détecter barres verticales
        bars = []
        for c in range(w):
            col = inp[:, c]
            non_bg = np.where(col != 0)[0]
            if len(non_bg) > 0 and len(non_bg) == (max(non_bg) - min(non_bg) + 1):
                bars.append({"col": c, "height": len(non_bg), "rows": non_bg})
        if not bars:
            return None
        # Trier par hauteur décroissante
        bars_sorted = sorted(bars, key=lambda b: b["height"], reverse=True)
        out = inp.copy()
        for rank, b in enumerate(bars_sorted):
            color = rank + 1
            for r in b["rows"]:
                out[r, b["col"]] = color
        return out

    all_ok = True
    for p in task.train_pairs:
        pred = solve(p["input"], task.train_pairs)
        if not np.array_equal(pred, p["output"]):
            all_ok = False
            break
    print(f"Test 08ed6ac7 (Tri barres par hauteur) : {'REUSSI' if all_ok else 'ECHEC'}")
    pred_test = solve(task.test_pairs[0]["input"], task.train_pairs)
    match_test = np.array_equal(pred_test, task.test_pairs[0]["output"])
    print(f"Test 08ed6ac7 (Exact match sur test) : {match_test}")

def test_primitive_05269061():
    task = ArcTask.load_from_file("Atelier_ARC_Prize/training/05269061.json")
    def solve(inp, ctx):
        h, w = inp.shape
        # Trouver la période P diagonale (r + c) % P
        best_p = None
        best_map = None
        for P in range(2, 6):
            mapping = {}
            valid = True
            for r in range(h):
                for c in range(w):
                    val = inp[r, c]
                    if val != 0:
                        mod = (r + c) % P
                        if mod in mapping and mapping[mod] != val:
                            valid = False
                            break
                        mapping[mod] = val
                if not valid:
                    break
            if valid and len(mapping) == P:
                best_p = P
                best_map = mapping
                break
        if best_p is None:
            return None
        out = np.zeros((h, w), dtype=np.int32)
        for r in range(h):
            for c in range(w):
                out[r, c] = best_map[(r + c) % best_p]
        return out

    all_ok = True
    for p in task.train_pairs:
        pred = solve(p["input"], task.train_pairs)
        if not np.array_equal(pred, p["output"]):
            all_ok = False
            break
    print(f"Test 05269061 (Pavage diagonal périodique) : {'REUSSI' if all_ok else 'ECHEC'}")
    pred_test = solve(task.test_pairs[0]["input"], task.train_pairs)
    match_test = np.array_equal(pred_test, task.test_pairs[0]["output"])
    print(f"Test 05269061 (Exact match sur test) : {match_test}")

test_primitive_0520fde7()
test_primitive_08ed6ac7()
test_primitive_05269061()
