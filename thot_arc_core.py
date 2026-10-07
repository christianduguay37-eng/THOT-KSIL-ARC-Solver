"""
==========================================================================
 🏛️ THOT ARC CORE : MOTEUR GÉOMÉTRIQUE & TOPOLOGIQUE DÉTERMINISTE
==========================================================================
Projet : ARC Prize (François Chollet / Kaggle)
Auteurs : Christian Duguay & Alix (Binôme Souverain)
Philosophie : Rigueur Déterministe, Invariance Topologique, Zéro-Flou
==========================================================================
"""

import json
import os
from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np

# Palette officielle ARC (10 couleurs standard)
ARC_COLORS = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "maroon"
}

class ArcGrid:
    """Représentation matricielle et opérations topologiques invariantes."""

    @staticmethod
    def to_np(grid: List[List[int]]) -> np.ndarray:
        return np.array(grid, dtype=np.int32)

    @staticmethod
    def to_list(arr: np.ndarray) -> List[List[int]]:
        return arr.tolist()

    # --- 1. Transformations Isométriques (D4 - Groupe Diédral d'Ordre 8) ---

    @staticmethod
    def rotate_90(arr: np.ndarray) -> np.ndarray:
        return np.rot90(arr, -1)

    @staticmethod
    def rotate_180(arr: np.ndarray) -> np.ndarray:
        return np.rot90(arr, 2)

    @staticmethod
    def rotate_270(arr: np.ndarray) -> np.ndarray:
        return np.rot90(arr, 1)

    @staticmethod
    def flip_h(arr: np.ndarray) -> np.ndarray:
        return np.fliplr(arr)

    @staticmethod
    def flip_v(arr: np.ndarray) -> np.ndarray:
        return np.flipud(arr)

    @staticmethod
    def transpose(arr: np.ndarray) -> np.ndarray:
        return arr.T

    @classmethod
    def get_all_d4_transforms(cls) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        return {
            "identity": lambda x: x.copy(),
            "rot90": cls.rotate_90,
            "rot180": cls.rotate_180,
            "rot270": cls.rotate_270,
            "flip_h": cls.flip_h,
            "flip_v": cls.flip_v,
            "transpose": cls.transpose,
            "anti_transpose": lambda x: np.flipud(np.fliplr(x.T))
        }

    # --- 2. Topologie & Détection de Cavités Fermées (Trous) ---

    @staticmethod
    def find_enclosed_holes(arr: np.ndarray, background: int = 0) -> np.ndarray:
        """
        Détecte les cellules d'arrière-plan (0) qui sont topologiquement
        enfermées par une frontière et ne peuvent PAS toucher les bords extérieurs.
        Renvoie un masque booléen des trous enfermés.
        """
        h, w = arr.shape
        visited = np.zeros((h, w), dtype=bool)
        queue = []

        # Identifier tous les points de bord qui sont 'background'
        for r in range(h):
            for c in [0, w - 1]:
                if arr[r, c] == background and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))
        for c in range(w):
            for r in [0, h - 1]:
                if arr[r, c] == background and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))

        # Propagation 4-connexe depuis l'extérieur (Flood fill extérieur)
        idx = 0
        while idx < len(queue):
            r, c = queue[idx]
            idx += 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if arr[nr, nc] == background and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

        # Les cavités enfermées sont les cellules background non visitées
        enclosed_mask = (arr == background) & (~visited)
        return enclosed_mask

    # --- 3. Détection de Composantes Connexes (Objets) ---

    @staticmethod
    def get_connected_components(arr: np.ndarray, background: int = 0, connectivity: int = 4) -> List[Dict[str, Any]]:
        """
        Extrait les objets connexes (même couleur, hors background).
        Renvoie une liste d'objets avec : couleur, masque, boîte englobante, aire.
        """
        h, w = arr.shape
        visited = np.zeros((h, w), dtype=bool)
        objects = []
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else [
            (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
        ]

        for r in range(h):
            for c in range(w):
                color = arr[r, c]
                if color == background or visited[r, c]:
                    continue

                # Nouvel objet trouvé
                coords = []
                queue = [(r, c)]
                visited[r, c] = True

                idx = 0
                while idx < len(queue):
                    cr, cc = queue[idx]
                    idx += 1
                    coords.append((cr, cc))

                    for dr, dc in deltas:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if not visited[nr, nc] and arr[nr, nc] == color:
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                rows = [p[0] for p in coords]
                cols = [p[1] for p in coords]
                min_r, max_r = min(rows), max(rows)
                min_c, max_c = min(cols), max(cols)

                obj_mask = np.zeros((h, w), dtype=bool)
                for pr, pc in coords:
                    obj_mask[pr, pc] = True

                sub_grid = arr[min_r:max_r+1, min_c:max_c+1].copy()

                objects.append({
                    "color": int(color),
                    "area": len(coords),
                    "coords": coords,
                    "bbox": (min_r, min_c, max_r - min_r + 1, max_c - min_c + 1),
                    "sub_grid": sub_grid,
                    "mask": obj_mask
                })

        return objects

    # --- 4. Rognage & Cadrage (Bounding Box) ---

    @staticmethod
    def crop_to_content(arr: np.ndarray, background: int = 0) -> np.ndarray:
        """Rogner la grille pour ne garder que la boîte englobante du contenu non-background."""
        non_bg = np.argwhere(arr != background)
        if len(non_bg) == 0:
            return arr.copy()
        min_r, min_c = non_bg.min(axis=0)
        max_r, max_c = non_bg.max(axis=0)
        return arr[min_r:max_r+1, min_c:max_c+1].copy()

    # --- 5. Produit de Kronecker & Auto-Similarité Fractale ---

    @staticmethod
    def kronecker_fractal(arr: np.ndarray, background: int = 0) -> np.ndarray:
        """
        Chaque pixel de couleur != background est remplacé par la matrice arr entière.
        Chaque pixel background est remplacé par une matrice de zéros de la taille de arr.
        """
        h, w = arr.shape
        out = np.zeros((h * h, w * w), dtype=np.int32)
        for r in range(h):
            for c in range(w):
                if arr[r, c] != background:
                    out[r*h:(r+1)*h, c*w:(c+1)*w] = arr
        return out


class ArcTask:
    """Modèle d'une tâche ARC avec vérification déterministe."""

    def __init__(self, task_id: str, data: Dict[str, Any]):
        self.task_id = task_id
        self.train_pairs = [
            {"input": ArcGrid.to_np(p["input"]), "output": ArcGrid.to_np(p["output"])}
            for p in data["train"]
        ]
        self.test_pairs = [
            {
                "input": ArcGrid.to_np(p["input"]),
                "output": ArcGrid.to_np(p["output"]) if "output" in p else None
            }
            for p in data["test"]
        ]

    @classmethod
    def load_from_file(cls, filepath: str) -> "ArcTask":
        task_id = os.path.basename(filepath).replace(".json", "")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(task_id, data)


class ThotArcSolver:
    """Moteur de déduction géométrique et synthèse de primitives invariantes."""

    def __init__(self):
        self.primitives: List[Tuple[str, Callable[[np.ndarray, List[Dict[str, np.ndarray]]], Optional[np.ndarray]]]] = []
        self._register_primitives()

    def _register_primitives(self):
        """Enregistre les règles et transformations candidates de THOT."""

        # 1. Isométries directes
        for name, func in ArcGrid.get_all_d4_transforms().items():
            self.primitives.append((f"iso_{name}", lambda inp, ctx, f=func: f(inp)))

        # 2. Kronecker Fractal (Auto-similarité)
        self.primitives.append((
            "fractal_kronecker",
            lambda inp, ctx: ArcGrid.kronecker_fractal(inp, background=0)
        ))

        # 3. Remplissage topologique de cavités fermées (Trous)
        def solve_enclosed_holes(inp: np.ndarray, ctx: List[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
            # Chercher quelle couleur de remplissage est utilisée dans le train
            hole_mask = ArcGrid.find_enclosed_holes(inp, background=0)
            if not np.any(hole_mask):
                return None
            # Déduire la couleur de remplissage depuis le contexte train
            fill_colors = []
            for pair in ctx:
                mask_tr = ArcGrid.find_enclosed_holes(pair["input"], background=0)
                if np.any(mask_tr) and pair["output"].shape == pair["input"].shape:
                    unique_vals = np.unique(pair["output"][mask_tr])
                    if len(unique_vals) == 1 and unique_vals[0] != 0:
                        fill_colors.append(int(unique_vals[0]))
            if fill_colors:
                target_color = max(set(fill_colors), key=fill_colors.count)
                out = inp.copy()
                out[hole_mask] = target_color
                return out
            return None

        self.primitives.append(("fill_enclosed_holes", solve_enclosed_holes))

        # 4. Rognage au contenu non-background (Crop to Bounding Box)
        self.primitives.append((
            "crop_bounding_box",
            lambda inp, ctx: ArcGrid.crop_to_content(inp, background=0)
        ))

        # 5. Remplacement de couleur direct (Palette Mapping 1-to-1)
        def solve_palette_mapping(inp: np.ndarray, ctx: List[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
            # Si dimensions identiques sur tous les trains
            if not all(p["input"].shape == p["output"].shape for p in ctx):
                return None
            # Construire la table de mapping
            mapping = {}
            for p in ctx:
                in_flat = p["input"].flatten()
                out_flat = p["output"].flatten()
                for u_in, u_out in zip(in_flat, out_flat):
                    if u_in in mapping and mapping[u_in] != u_out:
                        return None # Conflit, ce n'est pas un mapping 1-to-1 pur
                    mapping[u_in] = u_out
            # Appliquer le mapping
            out = inp.copy()
            for k, v in mapping.items():
                out[inp == k] = v
            return out

        self.primitives.append(("color_mapping_1to1", solve_palette_mapping))

        # 6. Extraction de l'objet le plus grand / le plus fréquent
        def solve_extract_largest_object(inp: np.ndarray, ctx: List[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
            objs = ArcGrid.get_connected_components(inp, background=0)
            if not objs:
                return None
            largest = max(objs, key=lambda o: o["area"])
            return largest["sub_grid"]

        self.primitives.append(("extract_largest_object", solve_extract_largest_object))

        # 7. Symétrie miroir avec complétion
        def solve_mirror_completion_h(inp: np.ndarray, ctx: List[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
            # Compléter la moitié gauche sur la droite ou inversement
            h, w = inp.shape
            mid = w // 2
            left = inp[:, :mid]
            right = inp[:, mid + (1 if w % 2 != 0 else 0):]
            # Si une moitié est vide et l'autre pleine
            out = inp.copy()
            if np.all(right == 0) and not np.all(left == 0):
                out[:, mid + (1 if w % 2 != 0 else 0):] = np.fliplr(left)
                return out
            if np.all(left == 0) and not np.all(right == 0):
                out[:, :mid] = np.fliplr(right)
                return out
            return None

        self.primitives.append(("mirror_completion_h", solve_mirror_completion_h))

        # 8. Intersection booléenne de volets séparés par une ligne uniforme (Split & Intersect)
        def solve_split_and_intersect(inp: np.ndarray, ctx: List[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
            h, w = inp.shape
            # Séparateur vertical
            sep_cols = [c for c in range(w) if len(np.unique(inp[:, c])) == 1 and inp[0, c] != 0]
            if sep_cols:
                c_sep = sep_cols[0]
                left = inp[:, :c_sep]
                right = inp[:, c_sep+1:]
                if left.shape == right.shape:
                    overlap = (left != 0) & (right != 0)
                    for p in ctx:
                        if p["output"].shape == left.shape:
                            vals = p["output"][p["output"] != 0]
                            out_col = int(vals[0]) if len(vals) > 0 else 2
                            out = np.zeros_like(left)
                            out[overlap] = out_col
                            return out

            # Séparateur horizontal
            sep_rows = [r for r in range(h) if len(np.unique(inp[r, :])) == 1 and inp[r, 0] != 0]
            if sep_rows:
                r_sep = sep_rows[0]
                top = inp[:r_sep, :]
                bottom = inp[r_sep+1:, :]
                if top.shape == bottom.shape:
                    overlap = (top != 0) & (bottom != 0)
                    for p in ctx:
                        if p["output"].shape == top.shape:
                            vals = p["output"][p["output"] != 0]
                            out_col = int(vals[0]) if len(vals) > 0 else 2
                            out = np.zeros_like(top)
                            out[overlap] = out_col
                            return out
            return None

        self.primitives.append(("split_and_intersect", solve_split_and_intersect))

        # 9. Tri et recoloration des barres par taille / hauteur
        def solve_rank_and_recolor_bars(inp: np.ndarray, ctx: List[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
            h, w = inp.shape
            bars = []
            for c in range(w):
                col = inp[:, c]
                non_bg = np.where(col != 0)[0]
                if len(non_bg) > 0 and len(non_bg) == (max(non_bg) - min(non_bg) + 1):
                    bars.append({"col": c, "height": len(non_bg), "rows": non_bg})
            if not bars:
                return None
            bars_sorted = sorted(bars, key=lambda b: b["height"], reverse=True)
            out = inp.copy()
            for rank, b in enumerate(bars_sorted):
                color = rank + 1
                for r in b["rows"]:
                    out[r, b["col"]] = color
            return out

        self.primitives.append(("rank_and_recolor_bars", solve_rank_and_recolor_bars))

        # 10. Pavage diagonal périodique
        def solve_periodic_diagonal_tiling(inp: np.ndarray, ctx: List[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
            h, w = inp.shape
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
                    out = np.zeros((h, w), dtype=np.int32)
                    for r in range(h):
                        for c in range(w):
                            out[r, c] = mapping[(r + c) % P]
                    return out
            return None

        self.primitives.append(("periodic_diagonal_tiling", solve_periodic_diagonal_tiling))

    def solve(self, task: ArcTask) -> Dict[str, Any]:
        """
        Éprouve chaque primitive déterministe sur 100% des paires de train.
        Si une primitive réussit sans aucune exception sur le train,
        elle est déclarée RÈGLE MAÎTRESSE et appliquée au test.
        """
        for prim_name, prim_fn in self.primitives:
            all_train_pass = True
            for train_pair in task.train_pairs:
                try:
                    pred = prim_fn(train_pair["input"], task.train_pairs)
                    if pred is None or not np.array_equal(pred, train_pair["output"]):
                        all_train_pass = False
                        break
                except Exception:
                    all_train_pass = False
                    break

            if all_train_pass:
                # Trouvé ! Appliquer au test
                test_predictions = []
                test_exact_matches = []
                for test_pair in task.test_pairs:
                    try:
                        pred_test = prim_fn(test_pair["input"], task.train_pairs)
                        test_predictions.append(ArcGrid.to_list(pred_test) if pred_test is not None else None)
                        if test_pair["output"] is not None and pred_test is not None:
                            match = bool(np.array_equal(pred_test, test_pair["output"]))
                            test_exact_matches.append(match)
                    except Exception:
                        test_predictions.append(None)
                        test_exact_matches.append(False)

                return {
                    "solved": True,
                    "rule": prim_name,
                    "task_id": task.task_id,
                    "test_pass": all(test_exact_matches) if test_exact_matches else None,
                    "predictions": test_predictions
                }

        return {
            "solved": False,
            "rule": None,
            "task_id": task.task_id,
            "test_pass": False,
            "predictions": []
        }
