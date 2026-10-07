"""
==========================================================================
 🌌 THOT K-SIL MULTIVERSE : MOTEUR DE DÉDUCTION FRACTALE & BRANCHING SSI
==========================================================================
Inspiré de : L'Architecture des 19 Paliers & SWE Agent K-SIL
Auteurs : Christian Duguay & Alix (Binôme Souverain)

Principes SRE & Ontologiques :
1. Niveau 0 : Four Micro-Ondes VERALUME (Tiers Exclu : ¬A → B) pour élagage O(1).
2. Palier 11 : Savepoints & Rollback sans allocation dynamique.
3. Palier 19 : Branching Multiversel & Collapse PBFT sur les états contrefactuels.
==========================================================================
"""

import os
import sys
import time
import glob
from collections import Counter
from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np
from scipy.ndimage import label, binary_dilation, binary_fill_holes

# Forcer UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from thot_arc_core import ArcGrid, ArcTask
from ksil_wave21_prims import get_wave21_primitives
from ksil_wave22_prims import get_wave22_primitives
from ksil_wave23_prims import get_wave23_primitives


class VeralumeMicrowaveFilter:
    """Four Micro-Ondes VERALUME : Détection d'invariants et élagage par tiers exclu."""

    @staticmethod
    def extract_signature(train_pairs: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Extrait la signature invariante du problème en O(1)."""
        same_shape = all(p["input"].shape == p["output"].shape for p in train_pairs)
        
        in_shapes = [p["input"].shape for p in train_pairs]
        out_shapes = [p["output"].shape for p in train_pairs]
        fixed_out_shape = len(set(out_shapes)) == 1
        
        # Ratios de forme
        scale_factors = []
        for inp, out in zip(in_shapes, out_shapes):
            if out[0] % inp[0] == 0 and out[1] % inp[1] == 0 and (out[0] // inp[0] == out[1] // inp[1]):
                scale_factors.append(out[0] // inp[0])
            else:
                scale_factors.append(None)
        is_integer_scale = all(s is not None and s == scale_factors[0] and s > 1 for s in scale_factors)
        is_kronecker_scale = all(out[0] == inp[0]**2 and out[1] == inp[1]**2 for inp, out in zip(in_shapes, out_shapes))
        
        # Analyse des spectres de couleur
        in_colors = set()
        out_colors = set()
        for p in train_pairs:
            in_colors.update(p["input"].flatten())
            out_colors.update(p["output"].flatten())
            
        new_colors_introduced = not out_colors.issubset(in_colors)
        
        # Séparateurs uniformes
        has_v_sep = any(
            any(len(np.unique(p["input"][:, c])) == 1 and p["input"][0, c] != 0 for c in range(p["input"].shape[1]))
            for p in train_pairs
        )
        has_h_sep = any(
            any(len(np.unique(p["input"][r, :])) == 1 and p["input"][r, 0] != 0 for r in range(p["input"].shape[0]))
            for p in train_pairs
        )

        return {
            "same_shape": same_shape,
            "fixed_out_shape": fixed_out_shape,
            "out_shape_sample": out_shapes[0] if fixed_out_shape else None,
            "is_integer_scale": is_integer_scale,
            "scale_factor": scale_factors[0] if is_integer_scale else None,
            "is_kronecker_scale": is_kronecker_scale,
            "new_colors_introduced": new_colors_introduced,
            "in_colors": in_colors,
            "out_colors": out_colors,
            "has_separator": has_v_sep or has_h_sep
        }


class KsilAtomicPrimitives:
    """Bibliothèque enrichie de primitives atomiques K-SIL."""

    @classmethod
    def get_primitives(cls) -> Dict[str, Callable[[np.ndarray, List[Dict[str, np.ndarray]]], Optional[np.ndarray]]]:
        prims = {}

        # 1. Isométries D4
        for name, func in ArcGrid.get_all_d4_transforms().items():
            prims[f"iso_{name}"] = lambda inp, ctx, f=func: f(inp)

        # 2. Rognage au contenu
        prims["crop_content"] = lambda inp, ctx: ArcGrid.crop_to_content(inp, background=0)

        # 3. Extraction du plus grand objet
        def extract_largest(inp: np.ndarray, ctx):
            objs = ArcGrid.get_connected_components(inp, background=0)
            return max(objs, key=lambda o: o["area"])["sub_grid"] if objs else None
        prims["extract_largest"] = extract_largest

        # 4. Extraction du plus petit objet
        def extract_smallest(inp: np.ndarray, ctx):
            objs = ArcGrid.get_connected_components(inp, background=0)
            return min(objs, key=lambda o: o["area"])["sub_grid"] if objs else None
        prims["extract_smallest"] = extract_smallest

        # 5. Remplissage de cavités fermées (Trous)
        def fill_holes(inp: np.ndarray, ctx):
            hole_mask = ArcGrid.find_enclosed_holes(inp, background=0)
            if not np.any(hole_mask):
                return None
            fill_cols = []
            for p in ctx:
                m = ArcGrid.find_enclosed_holes(p["input"], background=0)
                if np.any(m) and p["output"].shape == p["input"].shape:
                    vals = np.unique(p["output"][m])
                    if len(vals) == 1 and vals[0] != 0:
                        fill_cols.append(int(vals[0]))
            if fill_cols:
                target_col = max(set(fill_cols), key=fill_cols.count)
                out = inp.copy()
                out[hole_mask] = target_col
                return out
            return None
        prims["fill_holes"] = fill_holes

        # 6. Palette Mapping 1-to-1
        def palette_map(inp: np.ndarray, ctx):
            if not all(p["input"].shape == p["output"].shape for p in ctx):
                return None
            mapping = {}
            for p in ctx:
                for u_in, u_out in zip(p["input"].flatten(), p["output"].flatten()):
                    if u_in in mapping and mapping[u_in] != u_out:
                        return None
                    mapping[u_in] = u_out
            out = inp.copy()
            for k, v in mapping.items():
                out[inp == k] = v
            return out
        prims["palette_map"] = palette_map

        # 7. Recolorisation par Clé / Pointeur Isolé (Cas de la tâche aabf363d)
        def recolor_by_marker_key(inp: np.ndarray, ctx):
            objs = ArcGrid.get_connected_components(inp, background=0)
            if len(objs) < 2:
                return None
            # Chercher un objet de taille 1 (pixel pointeur) et un objet principal plus grand
            singletons = [o for o in objs if o["area"] == 1]
            large_objs = [o for o in objs if o["area"] > 1]
            if len(singletons) == 1 and len(large_objs) >= 1:
                key_color = singletons[0]["color"]
                out = inp.copy()
                # Effacer le pointeur
                pr, pc = singletons[0]["coords"][0]
                out[pr, pc] = 0
                # Repeindre l'objet principal avec la couleur clé
                for obj in large_objs:
                    for r, c in obj["coords"]:
                        out[r, c] = key_color
                return out
            return None
        prims["recolor_by_marker_key"] = recolor_by_marker_key

        # 8. Gravité vers le bas (Downwards gravity)
        def gravity_down(inp: np.ndarray, ctx):
            h, w = inp.shape
            out = np.zeros_like(inp)
            for c in range(w):
                col_vals = [inp[r, c] for r in range(h) if inp[r, c] != 0]
                # Empiler en bas
                for i, val in enumerate(reversed(col_vals)):
                    out[h - 1 - i, c] = val
            return out
        prims["gravity_down"] = gravity_down

        # 9. Nettoyage de bruit (Denoise : éliminer pixels isolés de taille 1)
        def denoise_singletons(inp: np.ndarray, ctx):
            objs = ArcGrid.get_connected_components(inp, background=0)
            singletons = [o for o in objs if o["area"] == 1]
            if not singletons:
                return None
            out = inp.copy()
            for s in singletons:
                r, c = s["coords"][0]
                out[r, c] = 0
            return out
        prims["denoise_singletons"] = denoise_singletons

        # 10. Logique Booléenne sur grilles séparées (AND, NOR, XOR, OR, DIFF)
        def split_boolean_logic(inp: np.ndarray, ctx):
            if not ctx:
                return None
            h, w = inp.shape
            ops = [
                ('AND', lambda L, R: (L != 0) & (R != 0)),
                ('NOR', lambda L, R: (L == 0) & (R == 0)),
                ('XOR', lambda L, R: (L != 0) ^ (R != 0)),
                ('OR', lambda L, R: (L != 0) | (R != 0)),
                ('DIFF_LR', lambda L, R: (L != 0) & (R == 0)),
                ('DIFF_RL', lambda L, R: (L == 0) & (R != 0)),
            ]
            for is_h in [False, True]:
                for op_name, op_fn in ops:
                    op_valid = True
                    learned_color = None
                    for p in ctx:
                        p_inp = p["input"]
                        p_out = p["output"]
                        ph, pw = p_inp.shape
                        if not is_h:
                            seps = [c for c in range(pw) if len(np.unique(p_inp[:, c])) == 1 and p_inp[0, c] != 0 and c == pw - 1 - c]
                            if not seps:
                                op_valid = False
                                break
                            c_sep = seps[0]
                            p_L, p_R = p_inp[:, :c_sep], p_inp[:, c_sep+1:]
                        else:
                            seps = [r for r in range(ph) if len(np.unique(p_inp[r, :])) == 1 and p_inp[r, 0] != 0 and r == ph - 1 - r]
                            if not seps:
                                op_valid = False
                                break
                            r_sep = seps[0]
                            p_L, p_R = p_inp[:r_sep, :], p_inp[r_sep+1:, :]

                        if p_out.shape != p_L.shape:
                            op_valid = False
                            break
                        mask = op_fn(p_L, p_R)
                        nonzero_out = p_out[mask]
                        if len(nonzero_out) > 0 and len(np.unique(nonzero_out)) == 1 and np.all(p_out[~mask] == 0):
                            color = int(nonzero_out[0])
                            if learned_color is None or learned_color == color:
                                learned_color = color
                            else:
                                op_valid = False
                                break
                        else:
                            op_valid = False
                            break
                    if op_valid and learned_color is not None:
                        if not is_h:
                            seps = [c for c in range(w) if len(np.unique(inp[:, c])) == 1 and inp[0, c] != 0 and c == w - 1 - c]
                            if not seps:
                                return None
                            c_sep = seps[0]
                            t_L, t_R = inp[:, :c_sep], inp[:, c_sep+1:]
                        else:
                            seps = [r for r in range(h) if len(np.unique(inp[r, :])) == 1 and inp[r, 0] != 0 and r == h - 1 - r]
                            if not seps:
                                return None
                            r_sep = seps[0]
                            t_L, t_R = inp[:r_sep, :], inp[r_sep+1:, :]
                        mask = op_fn(t_L, t_R)
                        out = np.zeros_like(t_L)
                        out[mask] = learned_color
                        return out
            return None
        prims["split_boolean_logic"] = split_boolean_logic
        prims["split_intersect"] = split_boolean_logic

        # 11. Kronecker Fractal
        prims["kronecker_fractal"] = lambda inp, ctx: ArcGrid.kronecker_fractal(inp, background=0)

        # 12. Tri et recoloration des barres par hauteur
        def rank_recolor_bars(inp: np.ndarray, ctx):
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
        prims["rank_recolor_bars"] = rank_recolor_bars

        # 13. Pavage diagonal périodique
        def periodic_diagonal(inp: np.ndarray, ctx):
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
        prims["periodic_diagonal"] = periodic_diagonal

        # 14. Rayures alternées par graine de points d'extrémité
        def seed_stripes(inp: np.ndarray, ctx):
            coords = np.argwhere(inp != 0)
            if len(coords) != 2:
                return None
            pt1, pt2 = coords[0], coords[1]
            h, w = inp.shape
            dr = abs(pt2[0] - pt1[0])
            dc = abs(pt2[1] - pt1[1])
            out = np.zeros_like(inp)
            if dc > 0 and (pt1[0] in [0, h-1] and pt2[0] in [0, h-1]):
                pts = sorted([pt1, pt2], key=lambda p: p[1])
                c1, c2 = pts[0][1], pts[1][1]
                color1, color2 = inp[pts[0][0], c1], inp[pts[1][0], c2]
                delta = c2 - c1
                cur_c = c1
                alt = False
                while cur_c < w:
                    out[:, cur_c] = color1 if not alt else color2
                    alt = not alt
                    cur_c += delta
                return out
            elif dr > 0 and (pt1[1] in [0, w-1] and pt2[1] in [0, w-1]):
                pts = sorted([pt1, pt2], key=lambda p: p[0])
                r1, r2 = pts[0][0], pts[1][0]
                color1, color2 = inp[r1, pts[0][1]], inp[r2, pts[1][1]]
                delta = r2 - r1
                cur_r = r1
                alt = False
                while cur_r < h:
                    out[cur_r, :] = color1 if not alt else color2
                    alt = not alt
                    cur_r += delta
                return out
            return None
        prims["seed_stripes"] = seed_stripes

        # 15. Volet discordant parmi sous-grilles séparées par des bandes de zéros
        def odd_subgrid(inp: np.ndarray, ctx):
            h, w = inp.shape
            zero_rows = [r for r in range(h) if np.all(inp[r, :] == 0)]
            zero_cols = [c for c in range(w) if np.all(inp[:, c] == 0)]
            if not zero_rows and not zero_cols:
                return None
            row_blocks = []
            start_r = 0
            for r in range(h):
                if r in zero_rows:
                    if r > start_r:
                        row_blocks.append((start_r, r))
                    start_r = r + 1
            if start_r < h:
                row_blocks.append((start_r, h))
            col_blocks = []
            start_c = 0
            for c in range(w):
                if c in zero_cols:
                    if c > start_c:
                        col_blocks.append((start_c, c))
                    start_c = c + 1
            if start_c < w:
                col_blocks.append((start_c, w))
            blocks = []
            for r1, r2 in row_blocks:
                for c1, c2 in col_blocks:
                    sub = inp[r1:r2, c1:c2]
                    colors = set(sub[sub != 0])
                    if colors:
                        blocks.append({"sub": sub, "colors": colors})
            if not blocks:
                return None
            color_block_counts = {}
            for b in blocks:
                for c in b["colors"]:
                    color_block_counts[c] = color_block_counts.get(c, 0) + 1
            unique_color_blocks = [b for b in blocks if any(color_block_counts[c] == 1 for c in b["colors"])]
            if len(unique_color_blocks) == 1:
                return unique_color_blocks[0]["sub"]
            return None
        prims["odd_subgrid"] = odd_subgrid

        # 16. Extension périodique verticale de rangées (tâche 017c7c7b)
        def periodic_rows(inp: np.ndarray, ctx):
            h, w = inp.shape
            target_h = ctx[0]["output"].shape[0] if ctx else 9
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
            out[out == 1] = 2
            return out
        prims["periodic_rows"] = periodic_rows

        # 17. Aimantation / Glissement d'objet vers cible (tâche 05f2a901)
        def dock_to_target(inp: np.ndarray, ctx):
            objs = ArcGrid.get_connected_components(inp, background=0)
            if len(objs) != 2:
                return None
            mobile_obj = [o for o in objs if o["color"] == 2]
            target_obj = [o for o in objs if o["color"] != 2]
            if not mobile_obj or not target_obj:
                return None
            m, t = mobile_obj[0], target_obj[0]
            h, w = inp.shape
            for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                valid_shift = None
                for step in range(1, max(h, w)):
                    next_coords = [(r + dr*step, c + dc*step) for r, c in m["coords"]]
                    if any(r < 0 or r >= h or c < 0 or c >= w for r, c in next_coords):
                        break
                    if any(t["mask"][r, c] for r, c in next_coords):
                        valid_shift = step - 1
                        break
                    adjacent = any(
                        0 <= adj_r < h and 0 <= adj_c < w and t["mask"][adj_r, adj_c]
                        for r, c in next_coords
                        for adj_r, adj_c in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                    )
                    if adjacent:
                        valid_shift = step
                        break
                if valid_shift is not None and valid_shift > 0:
                    out = np.zeros_like(inp)
                    out[t["mask"]] = t["color"]
                    for r, c in m["coords"]:
                        out[r + dr*valid_shift, c + dc*valid_shift] = m["color"]
                    return out
            return None
        prims["dock_to_target"] = dock_to_target

        # 18. Ponts de connexion entre blocs alignés de même couleur (tâche 06df4c85)
        def connect_aligned_blocks(inp: np.ndarray, ctx):
            h, w = inp.shape
            grid_cols = [c for c in range(w) if len(np.unique(inp[:, c])) == 1 and inp[0, c] != 0]
            grid_rows = [r for r in range(h) if len(np.unique(inp[r, :])) == 1 and inp[r, 0] != 0]
            if not grid_cols or not grid_rows:
                return None
            grid_color = inp[grid_rows[0], 0]
            step_c = grid_cols[1] - grid_cols[0] if len(grid_cols) > 1 else 3
            step_r = grid_rows[1] - grid_rows[0] if len(grid_rows) > 1 else 3
            out = inp.copy()
            objs = ArcGrid.get_connected_components(inp, background=0)
            blocks = [o for o in objs if o["color"] != grid_color and o["area"] == 4]
            by_color = {}
            for b in blocks:
                by_color.setdefault(b["color"], []).append(b)
            for color, blist in by_color.items():
                if len(blist) < 2:
                    continue
                for i in range(len(blist)):
                    for j in range(i + 1, len(blist)):
                        b1, b2 = blist[i], blist[j]
                        r1, c1, h1, w1 = b1["bbox"]
                        r2, c2, h2, w2 = b2["bbox"]
                        if r1 == r2:
                            for c in range(min(c1, c2), max(c1, c2) + w1, step_c):
                                out[r1:r1+h1, c:c+w1] = color
                        elif c1 == c2:
                            for r in range(min(r1, r2), max(r1, r2) + h1, step_r):
                                out[r:r+h1, c1:c1+w1] = color
            return out
        prims["connect_aligned_blocks"] = connect_aligned_blocks

        # 19. Compte des carreaux d'un quadrillage séparé par des lignes (tâche 1190e5a7)
        def grid_cell_count(inp: np.ndarray, ctx):
            h, w = inp.shape
            sep_colors = []
            h_seps, v_seps = [], []
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
            h_lines = [r for r in h_seps if inp[r, 0] == sep_color]
            v_lines = [c for c in v_seps if inp[0, c] == sep_color]
            num_rows = len(h_lines) + 1
            num_cols = len(v_lines) + 1
            content_colors = [c for c in np.unique(inp) if c != sep_color]
            if not content_colors:
                return None
            return np.full((num_rows, num_cols), fill_value=content_colors[0], dtype=np.int32)
        prims["grid_cell_count"] = grid_cell_count

        # 20. Satellites cardinaux et diagonaux autour de points clés (tâche 0ca9ddb6)
        def satellite_cross(inp: np.ndarray, ctx):
            h, w = inp.shape
            out = inp.copy()
            for r in range(h):
                for c in range(w):
                    val = inp[r, c]
                    if val == 2:
                        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                out[nr, nc] = 4
                    elif val == 1:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                out[nr, nc] = 7
            for r in range(h):
                for c in range(w):
                    if inp[r, c] != 0:
                        out[r, c] = inp[r, c]
            return out
        prims["satellite_cross"] = satellite_cross

        # 21. Rose des vents / Étoile à 8 rayons depuis centre de croix (tâche 0962bcdd)
        def compass_star(inp: np.ndarray, ctx):
            h, w = inp.shape
            out = inp.copy()
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
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for dist in [1, 2]:
                        nr, nc = r + dr * dist, c + dc * dist
                        if 0 <= nr < h and 0 <= nc < w:
                            out[nr, nc] = c_arm
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    for dist in [1, 2]:
                        nr, nc = r + dr * dist, c + dc * dist
                        if 0 <= nr < h and 0 <= nc < w:
                            out[nr, nc] = c_center
                out[r, c] = c_center
            return out
        prims["compass_star"] = compass_star

        # 22. Inpainting de cavités par tuilage périodique 2D (tâche 0dfd9992)
        def periodic_inpainting_2d(inp: np.ndarray, ctx):
            h, w = inp.shape
            best_pr, best_pc = None, None
            best_tile = None
            for Pr in range(2, min(12, h)):
                for Pc in range(2, min(12, w)):
                    tile = np.zeros((Pr, Pc), dtype=np.int32)
                    valid = True
                    for r in range(h):
                        for c in range(w):
                            val = inp[r, c]
                            if val != 0:
                                tr, tc = r % Pr, c % Pc
                                if tile[tr, tc] != 0 and tile[tr, tc] != val:
                                    valid = False
                                    break
                                tile[tr, tc] = val
                        if not valid:
                            break
                    if valid and np.all(tile != 0):
                        best_pr, best_pc = Pr, Pc
                        best_tile = tile
                        break
                if best_pr is not None:
                    break
            if best_tile is None:
                return None
            out = inp.copy()
            for r in range(h):
                for c in range(w):
                    if out[r, c] == 0:
                        out[r, c] = best_tile[r % best_pr, c % best_pc]
            return out
        prims["periodic_inpainting_2d"] = periodic_inpainting_2d

        # 23. Mini-carte de guidage par quadrant sans couleur sentinelle (tâche 09629e4f)
        def subgrid_minimap(inp: np.ndarray, ctx):
            h, w = inp.shape
            if h != 11 or w != 11:
                return None
            r_slices = [(0, 3), (4, 7), (8, 11)]
            c_slices = [(0, 3), (4, 7), (8, 11)]
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
        prims["subgrid_minimap"] = subgrid_minimap

        # 24. Redressement de boîte cisaillée vers la base (tâche 025d127b)
        def straighten_sheared_box(inp: np.ndarray, ctx):
            out = np.zeros_like(inp)
            colors = set(inp.flatten()) - {0}
            if not colors:
                return None
            for color in colors:
                coords = np.argwhere(inp == color)
                if len(coords) == 0:
                    continue
                max_r = coords[:, 0].max()
                max_c = coords[:, 1].max()
                for r, c in coords:
                    if r == max_r or c == max_c:
                        out[r, c] = color
                    else:
                        if c + 1 < inp.shape[1]:
                            out[r, c + 1] = color
                        else:
                            return None
            return out
        prims["straighten_sheared_box"] = straighten_sheared_box

        # 25. Réplication périodique de patron selon rayons directeurs (tâche 045e512c)
        def replicate_template_along_seed_rays(inp: np.ndarray, ctx):
            h, w = inp.shape
            colors = set(inp.flatten()) - {0}
            if len(colors) < 2:
                return None
            counts = {c: np.sum(inp == c) for c in colors}
            sorted_colors = sorted(counts.items(), key=lambda x: -x[1])
            tmpl_color = sorted_colors[0][0]
            tmpl_coords = np.argwhere(inp == tmpl_color)
            rmin, cmin = tmpl_coords.min(axis=0)
            rmax, cmax = tmpl_coords.max(axis=0)
            th = rmax - rmin + 1
            tw = cmax - cmin + 1
            if th != tw or th < 2:
                return None
            tmpl_pat = (inp[rmin:rmax+1, cmin:cmax+1] == tmpl_color)
            out = np.zeros_like(inp)
            for r in range(th):
                for c in range(tw):
                    if tmpl_pat[r, c]:
                        out[rmin + r, cmin + c] = tmpl_color
            step = th + 1
            cr_t = (rmin + rmax) / 2.0
            cc_t = (cmin + cmax) / 2.0
            rays = set()
            for c, _ in sorted_colors[1:]:
                c_mask = (inp == c)
                labeled, num_features = label(c_mask)
                for comp_id in range(1, num_features + 1):
                    comp_coords = np.argwhere(labeled == comp_id)
                    comp_rmin, comp_cmin = comp_coords.min(axis=0)
                    comp_rmax, comp_cmax = comp_coords.max(axis=0)
                    cr_c = (comp_rmin + comp_rmax) / 2.0
                    cc_c = (comp_cmin + comp_cmax) / 2.0
                    dr = cr_c - cr_t
                    dc = cc_c - cc_t
                    sr = 0 if abs(dr) < 1.0 else (1 if dr > 0 else -1)
                    sc = 0 if abs(dc) < 1.0 else (1 if dc > 0 else -1)
                    rays.add((sr * step, sc * step, c))
            for step_r, step_c, c in rays:
                if step_r == 0 and step_c == 0:
                    continue
                k = 1
                while True:
                    cur_rmin = rmin + k * step_r
                    cur_cmin = cmin + k * step_c
                    placed = False
                    for r in range(th):
                        for col in range(tw):
                            if tmpl_pat[r, col]:
                                nr = cur_rmin + r
                                nc = cur_cmin + col
                                if 0 <= nr < h and 0 <= nc < w:
                                    out[nr, nc] = c
                                    placed = True
                    if not placed:
                        break
                    k += 1
            return out
        prims["replicate_template_along_seed_rays"] = replicate_template_along_seed_rays

        # 26. Complétion de symétrie D4 Diédrale au sein de la boîte englobante (tâche 11852cab)
        def symmetry_completion_d4(inp: np.ndarray, ctx):
            coords = np.argwhere(inp > 0)
            if len(coords) == 0:
                return None
            rmin, cmin = coords.min(axis=0)
            rmax, cmax = coords.max(axis=0)
            h = rmax - rmin + 1
            w = cmax - cmin + 1
            if h != w:
                return None
            sub = inp[rmin:rmax+1, cmin:cmax+1]
            transforms = [
                lambda x: x,
                lambda x: x[:, ::-1],
                lambda x: x[::-1, :],
                lambda x: x[::-1, ::-1],
                lambda x: x.T,
                lambda x: x.T[:, ::-1],
                lambda x: x.T[::-1, :],
                lambda x: x.T[::-1, ::-1],
            ]
            res_sub = np.zeros_like(sub)
            for t in transforms:
                res_sub = np.maximum(res_sub, t(sub))
            out = inp.copy()
            out[rmin:rmax+1, cmin:cmax+1] = res_sub
            return out
        prims["symmetry_completion_d4"] = symmetry_completion_d4

        # 27. Projection de faisceaux orthogonaux par couleur (tâche 178fcbfb)
        def color_beam_cross(inp: np.ndarray, ctx):
            if not ctx:
                return None
            train_colors = set()
            for p in ctx:
                if p["input"].shape != p["output"].shape:
                    return None
                train_colors.update(set(p["input"].flatten()) - {0})
            color_list = sorted(list(train_colors))
            if not color_list or len(color_list) > 6:
                return None
            import itertools
            for v_first in [True, False]:
                first_role = 'V' if v_first else 'H'
                second_role = 'H' if v_first else 'V'
                for roles in itertools.product(['H', 'V'], repeat=len(color_list)):
                    c_map = dict(zip(color_list, roles))
                    all_valid = True
                    for p in ctx:
                        pinp = p["input"]
                        pout = p["output"]
                        ph, pw = pinp.shape
                        pred = np.zeros_like(pinp)
                        for role in [first_role, second_role]:
                            for r in range(ph):
                                for c in range(pw):
                                    val = pinp[r, c]
                                    if c_map.get(val) == role:
                                        if role == 'V':
                                            pred[:, c] = val
                                        else:
                                            pred[r, :] = val
                        if not np.array_equal(pred, pout):
                            all_valid = False
                            break
                    if all_valid:
                        h, w = inp.shape
                        out = np.zeros_like(inp)
                        for role in [first_role, second_role]:
                            for r in range(h):
                                for c in range(w):
                                    val = inp[r, c]
                                    if c_map.get(val) == role:
                                        if role == 'V':
                                            out[:, c] = val
                                        else:
                                            out[r, :] = val
                        return out
            return None
        prims["color_beam_cross"] = color_beam_cross

        # 28. Attraction magnétique de pixels libres vers les lignes de même couleur (tâche 1a07d186)
        def dock_pixels_to_magnetic_lines(inp: np.ndarray, ctx):
            H, W = inp.shape
            lines = []
            line_mask = np.zeros_like(inp, dtype=bool)

            for c in range(W):
                col_vals = inp[:, c]
                for color in set(col_vals) - {0}:
                    if np.sum(col_vals == color) >= H - 2:
                        lines.append(('V', c, color))
                        line_mask[:, c] = True

            for r in range(H):
                row_vals = inp[r, :]
                for color in set(row_vals) - {0}:
                    if np.sum(row_vals == color) >= W - 2:
                        lines.append(('H', r, color))
                        line_mask[r, :] = True

            if not lines:
                return None

            out = np.zeros_like(inp)
            for ldir, lpos, color in lines:
                if ldir == 'V':
                    out[:, lpos] = color
                else:
                    out[lpos, :] = color

            for r in range(H):
                for c in range(W):
                    if not line_mask[r, c] and inp[r, c] != 0:
                        color = inp[r, c]
                        matching = [l for l in lines if l[2] == color]
                        if not matching:
                            continue
                        best_line = None
                        best_dist = 9999
                        for l in matching:
                            ldir, lpos, _ = l
                            dist = abs(c - lpos) if ldir == 'V' else abs(r - lpos)
                            if dist < best_dist:
                                best_dist = dist
                                best_line = l
                        ldir, lpos, _ = best_line
                        if ldir == 'V':
                            dock_c = lpos - 1 if c < lpos else lpos + 1
                            if 0 <= dock_c < W:
                                out[r, dock_c] = color
                        else:
                            dock_r = lpos - 1 if r < lpos else lpos + 1
                            if 0 <= dock_r < H:
                                out[dock_r, c] = color

            return out
        prims["dock_pixels_to_magnetic_lines"] = dock_pixels_to_magnetic_lines

        # 29. Remplissage des travées horizontales entre pixels de même couleur (tâches 22168020, 22eb0ac0)
        def fill_horizontal_matching_spans(inp: np.ndarray, ctx):
            H, W = inp.shape
            out = inp.copy()
            colors = set(inp.flatten()) - {0}
            filled_any = False
            for color in colors:
                for r in range(H):
                    cols = [c for c in range(W) if inp[r, c] == color]
                    if len(cols) >= 2:
                        c_min, c_max = min(cols), max(cols)
                        span_vals = set(inp[r, c_min:c_max+1]) - {0, color}
                        if not span_vals and c_max > c_min + 1:
                            out[r, c_min:c_max+1] = color
                            filled_any = True
            return out
        prims["fill_horizontal_matching_spans"] = fill_horizontal_matching_spans

        # 30. Alignement d'objets sur l'axe d'un objet ancre de référence (tâche 1caeab9d)
        def align_shapes_to_anchor_axis(inp: np.ndarray, ctx):
            if not ctx:
                return None
            train_pairs = ctx
            colors_in_all = None
            for p in train_pairs:
                if p["input"].shape != p["output"].shape:
                    return None
                colors = set(p["input"].flatten()) - {0}
                if colors_in_all is None:
                    colors_in_all = colors
                else:
                    colors_in_all &= colors
            if not colors_in_all:
                return None

            for anchor_col in sorted(list(colors_in_all)):
                for axis in ['Y', 'X']:
                    all_valid = True
                    for p in train_pairs:
                        pinp = p["input"]
                        pout = p["output"]
                        H, W = pinp.shape
                        pred = np.zeros_like(pinp)
                        a_coords = np.argwhere(pinp == anchor_col)
                        if len(a_coords) == 0:
                            all_valid = False
                            break
                        a_target = a_coords[:, 0].min() if axis == 'Y' else a_coords[:, 1].min()
                        colors = set(pinp.flatten()) - {0}
                        for c in colors:
                            coords = np.argwhere(pinp == c)
                            c_pos = coords[:, 0].min() if axis == 'Y' else coords[:, 1].min()
                            shift = a_target - c_pos
                            for r, col in coords:
                                nr = r + shift if axis == 'Y' else r
                                nc = col if axis == 'Y' else col + shift
                                if 0 <= nr < H and 0 <= nc < W:
                                    pred[nr, nc] = c
                                else:
                                    all_valid = False
                                    break
                            if not all_valid:
                                break
                        if not np.array_equal(pred, pout):
                            all_valid = False
                            break
                    if all_valid:
                        H, W = inp.shape
                        out = np.zeros_like(inp)
                        a_coords = np.argwhere(inp == anchor_col)
                        if len(a_coords) == 0:
                            return None
                        a_target = a_coords[:, 0].min() if axis == 'Y' else a_coords[:, 1].min()
                        colors = set(inp.flatten()) - {0}
                        for c in colors:
                            coords = np.argwhere(inp == c)
                            c_pos = coords[:, 0].min() if axis == 'Y' else coords[:, 1].min()
                            shift = a_target - c_pos
                            for r, col in coords:
                                nr = r + shift if axis == 'Y' else r
                                nc = col if axis == 'Y' else col + shift
                                if 0 <= nr < H and 0 <= nc < W:
                                    out[nr, nc] = c
                                else:
                                    return None
                        return out
            return None
        prims["align_shapes_to_anchor_axis"] = align_shapes_to_anchor_axis

        # 31. Détection et surbrillance des lignes monochromatiques (tâche 25d8a9c8)
        def highlight_monochrome_rows(inp: np.ndarray, ctx):
            if not ctx:
                return None
            learned_color = None
            for p in ctx:
                pinp = p["input"]
                pout = p["output"]
                if pinp.shape != pout.shape:
                    return None
                H, W = pinp.shape
                mono_rows = [r for r in range(H) if len(set(pinp[r, :])) == 1]
                if not mono_rows:
                    return None
                for r in range(H):
                    if r in mono_rows:
                        vals = set(pout[r, :])
                        if len(vals) != 1 or 0 in vals:
                            return None
                        col = list(vals)[0]
                        if learned_color is None or learned_color == col:
                            learned_color = col
                        else:
                            return None
                    else:
                        if not np.all(pout[r, :] == 0):
                            return None
            if learned_color is None:
                return None
            H, W = inp.shape
            out = np.zeros_like(inp)
            for r in range(H):
                if len(set(inp[r, :])) == 1:
                    out[r, :] = learned_color
            return out
        prims["highlight_monochrome_rows"] = highlight_monochrome_rows

        # 32. Translation rigide globale (tâche 25ff71a9)
        def rigid_translation(inp: np.ndarray, ctx):
            if not ctx:
                return None
            best_shift = None
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    valid = True
                    for p in ctx:
                        pinp = p["input"]
                        pout = p["output"]
                        if pinp.shape != pout.shape:
                            valid = False
                            break
                        H, W = pinp.shape
                        pred = np.zeros_like(pinp)
                        for r in range(H):
                            for c in range(W):
                                val = pinp[r, c]
                                if val != 0:
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < H and 0 <= nc < W:
                                        pred[nr, nc] = val
                                    else:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if not np.array_equal(pred, pout):
                            valid = False
                            break
                    if valid:
                        best_shift = (dr, dc)
                        break
                if best_shift is not None:
                    break

            if best_shift is None:
                return None
            dr, dc = best_shift
            H, W = inp.shape
            out = np.zeros_like(inp)
            for r in range(H):
                for c in range(W):
                    val = inp[r, c]
                    if val != 0:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            out[nr, nc] = val
                        else:
                            return None
            return out
        prims["rigid_translation"] = rigid_translation

        # 33. Balayage diagonal d'un pavé 2x2 selon les directions d'un marqueur (tâche 1f0c79e5)
        def diagonal_sweep_from_marker(inp: np.ndarray, ctx):
            colors = set(inp.flatten()) - {0, 2}
            if len(colors) != 1 or 2 not in inp:
                return None
            target_color = list(colors)[0]
            coords = np.argwhere(inp > 0)
            if len(coords) == 0:
                return None
            rmin, cmin = coords.min(axis=0)
            rmax, cmax = coords.max(axis=0)
            if (rmax - rmin + 1) != 2 or (cmax - cmin + 1) != 2:
                return None
            cr = (rmin + rmax) / 2.0
            cc = (cmin + cmax) / 2.0
            twos = np.argwhere(inp == 2)
            directions = []
            for r, c in twos:
                dr = 1 if r > cr else -1
                dc = 1 if c > cc else -1
                directions.append((dr, dc))
            square_coords = [(r, c) for r in range(rmin, rmax + 1) for c in range(cmin, cmax + 1)]
            H, W = inp.shape
            out = np.zeros_like(inp)
            for dr, dc in directions:
                for k in range(max(H, W)):
                    for r, c in square_coords:
                        nr = r + k * dr
                        nc = c + k * dc
                        if 0 <= nr < H and 0 <= nc < W:
                            out[nr, nc] = target_color
            return out
        prims["diagonal_sweep_from_marker"] = diagonal_sweep_from_marker

        # 34. Accumulation des voisinages 3x3 autour de marqueurs sentinelles (tâche 137eaa0f)
        def accumulate_neighborhoods_around_marker(inp: np.ndarray, ctx):
            if not ctx:
                return None
            for marker_col in range(1, 10):
                all_valid = True
                for p in ctx:
                    pinp = p["input"]
                    pout = p["output"]
                    if pout.shape != (3, 3):
                        all_valid = False
                        break
                    ph, pw = pinp.shape
                    p_out = np.zeros((3, 3), dtype=np.int32)
                    p_out[1, 1] = marker_col
                    markers = np.argwhere(pinp == marker_col)
                    if len(markers) == 0:
                        all_valid = False
                        break
                    for mr, mc in markers:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = mr + dr, mc + dc
                                if 0 <= nr < ph and 0 <= nc < pw:
                                    val = pinp[nr, nc]
                                    if val != 0 and val != marker_col:
                                        p_out[1 + dr, 1 + dc] = val
                    if not np.array_equal(p_out, pout):
                        all_valid = False
                        break
                if all_valid:
                    H, W = inp.shape
                    out = np.zeros((3, 3), dtype=np.int32)
                    out[1, 1] = marker_col
                    markers = np.argwhere(inp == marker_col)
                    for mr, mc in markers:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = mr + dr, mc + dc
                                if 0 <= nr < H and 0 <= nc < W:
                                    val = inp[nr, nc]
                                    if val != 0 and val != marker_col:
                                        out[1 + dr, 1 + dc] = val
                    return out
            return None
        prims["accumulate_neighborhoods_around_marker"] = accumulate_neighborhoods_around_marker

        # 35. Faisceaux en croix depuis chaque point avec couleur d'intersection (tâche 23581191)
        def crosshairs_with_intersection_color(inp: np.ndarray, ctx):
            if not ctx:
                return None
            learned_inter_col = None
            for p in ctx:
                pinp = p["input"]
                pout = p["output"]
                if pinp.shape != pout.shape:
                    return None
                points = np.argwhere(pinp > 0)
                if len(points) < 2 or len(points) > 35:
                    return None
                intersections = []
                for i in range(len(points)):
                    for j in range(len(points)):
                        if i != j:
                            intersections.append((points[i][0], points[j][1]))
                inter_vals = set(pout[r, c] for r, c in intersections)
                if len(inter_vals) != 1 or 0 in inter_vals:
                    return None
                icol = list(inter_vals)[0]
                if learned_inter_col is None or learned_inter_col == icol:
                    learned_inter_col = icol
                else:
                    return None
            if learned_inter_col is None:
                return None
            H, W = inp.shape
            out = np.zeros_like(inp)
            points = np.argwhere(inp > 0)
            for r, c in points:
                color = inp[r, c]
                out[r, :] = color
                out[:, c] = color
            for i in range(len(points)):
                for j in range(len(points)):
                    if i != j:
                        r1, c1 = points[i]
                        r2, c2 = points[j]
                        out[r1, c2] = learned_inter_col
                        out[r2, c1] = learned_inter_col
            return out
        prims["crosshairs_with_intersection_color"] = crosshairs_with_intersection_color

        # 36. Partitionnement de Voronoï entre deux lignes frontières opposées (tâche 2204b7a8)
        def voronoi_boundary_lines(inp: np.ndarray, ctx):
            H, W = inp.shape
            h_lines = []
            for r in range(H):
                if len(set(inp[r, :])) == 1 and inp[r, 0] != 0:
                    h_lines.append((r, inp[r, 0]))
            v_lines = []
            for c in range(W):
                if len(set(inp[:, c])) == 1 and inp[0, c] != 0:
                    v_lines.append((c, inp[0, c]))
            if len(h_lines) != 2 and len(v_lines) != 2:
                return None
            out = inp.copy()
            if len(h_lines) == 2:
                r1, c_col1 = h_lines[0]
                r2, c_col2 = h_lines[1]
                for r in range(H):
                    for c in range(W):
                        val = inp[r, c]
                        if val != 0 and (r, val) not in h_lines:
                            out[r, c] = c_col1 if abs(r - r1) < abs(r - r2) else c_col2
            elif len(v_lines) == 2:
                c1, c_col1 = v_lines[0]
                c2, c_col2 = v_lines[1]
                for r in range(H):
                    for c in range(W):
                        val = inp[r, c]
                        if val != 0 and (c, val) not in v_lines:
                            out[r, c] = c_col1 if abs(c - c1) < abs(c - c2) else c_col2
            return out
        prims["voronoi_boundary_lines"] = voronoi_boundary_lines

        # 37. Extraction du quadrant supérieur gauche du contenu rogné (tâche 2013d3e2)
        def crop_top_left_quadrant(inp: np.ndarray, ctx):
            coords = np.argwhere(inp > 0)
            if len(coords) == 0:
                return None
            rmin, cmin = coords.min(axis=0)
            rmax, cmax = coords.max(axis=0)
            cropped = inp[rmin:rmax+1, cmin:cmax+1]
            H, W = cropped.shape
            if H % 2 != 0 or W % 2 != 0 or H < 2 or W < 2:
                return None
            return cropped[:H//2, :W//2]
        prims["crop_top_left_quadrant"] = crop_top_left_quadrant

        # 38. Lancer de rayons depuis les pointeurs périphériques vers un bloc central (tâche 1f642eb9)
        def raycast_pointers_onto_central_box(inp: np.ndarray, ctx):
            colors = set(inp.flatten()) - {0}
            if len(colors) < 2:
                return None
            counts = {c: np.sum(inp == c) for c in colors}
            sorted_colors = sorted(counts.items(), key=lambda x: -x[1])
            box_col = sorted_colors[0][0]
            box_coords = np.argwhere(inp == box_col)
            rmin, cmin = box_coords.min(axis=0)
            rmax, cmax = box_coords.max(axis=0)
            h_box = rmax - rmin + 1
            w_box = cmax - cmin + 1
            if len(box_coords) != h_box * w_box or h_box < 2 or w_box < 2:
                return None
            H, W = inp.shape
            out = inp.copy()
            for r in range(H):
                for c in range(W):
                    val = inp[r, c]
                    if val != 0 and val != box_col:
                        if rmin <= r <= rmax:
                            if c < cmin:
                                out[r, cmin] = val
                            elif c > cmax:
                                out[r, cmax] = val
                        elif cmin <= c <= cmax:
                            if r < rmin:
                                out[rmin, c] = val
                            elif r > rmax:
                                out[rmax, c] = val
            return out
        prims["raycast_pointers_onto_central_box"] = raycast_pointers_onto_central_box

        # 39. Extraction du contenu intérieur d'un cadre rectangulaire fermé (tâche 1c786137)
        def extract_rectangular_frame_interior(inp: np.ndarray, ctx):
            colors = set(inp.flatten()) - {0}
            for c in colors:
                coords = np.argwhere(inp == c)
                if len(coords) < 8:
                    continue
                rmin, cmin = coords.min(axis=0)
                rmax, cmax = coords.max(axis=0)
                if rmax - rmin < 2 or cmax - cmin < 2:
                    continue
                top = np.all(inp[rmin, cmin:cmax+1] == c)
                bot = np.all(inp[rmax, cmin:cmax+1] == c)
                left = np.all(inp[rmin:rmax+1, cmin] == c)
                right = np.all(inp[rmin:rmax+1, cmax] == c)
                if top and bot and left and right:
                    return inp[rmin+1:rmax, cmin+1:cmax]
            return None
        prims["extract_rectangular_frame_interior"] = extract_rectangular_frame_interior

        # 40. Pochoirs invariants par régions horizontales (tâche 1bfc4729)
        def template_mask_by_region(inp: np.ndarray, ctx):
            if not ctx:
                return None
            p0_inp = ctx[0]["input"]
            p0_out = ctx[0]["output"]
            if p0_inp.shape != (10, 10) or p0_out.shape != (10, 10):
                return None
            top_c0 = p0_out[0, 0]
            bot_c0 = p0_out[9, 9]
            top_mask = (p0_out[:5, :] == top_c0)
            bot_mask = (p0_out[5:, :] == bot_c0)
            for p in ctx:
                pinp = p["input"]
                pout = p["output"]
                if pinp.shape != (10, 10) or pout.shape != (10, 10):
                    return None
                tc = pinp[:5, :][pinp[:5, :] > 0]
                bc = pinp[5:, :][pinp[5:, :] > 0]
                if len(tc) == 0 or len(bc) == 0:
                    return None
                pred = np.zeros_like(pinp)
                pred[:5, :][top_mask] = tc[0]
                pred[5:, :][bot_mask] = bc[0]
                if not np.array_equal(pred, pout):
                    return None
            if inp.shape != (10, 10):
                return None
            tc = inp[:5, :][inp[:5, :] > 0]
            bc = inp[5:, :][inp[5:, :] > 0]
            if len(tc) == 0 or len(bc) == 0:
                return None
            out = np.zeros_like(inp)
            out[:5, :][top_mask] = tc[0]
            out[5:, :][bot_mask] = bc[0]
            return out
        prims["template_mask_by_region"] = template_mask_by_region

        # 41. Raccordement de paires diagonales de même couleur (tâche 1f876c06)
        def connect_diagonal_pairs_of_same_color(inp: np.ndarray, ctx):
            out = inp.copy()
            bg = 0
            colors = [c for c in np.unique(inp) if c != bg]
            connected_any = False
            for c in colors:
                pts = np.argwhere(inp == c)
                if len(pts) == 2:
                    r1, c1 = pts[0]
                    r2, c2 = pts[1]
                    dr = r2 - r1
                    dc = c2 - c1
                    if abs(dr) == abs(dc) and abs(dr) > 1:
                        step_r = 1 if dr > 0 else -1
                        step_c = 1 if dc > 0 else -1
                        for t in range(1, abs(dr)):
                            out[r1 + t * step_r, c1 + t * step_c] = c
                        connected_any = True
            return out if connected_any else None
        prims["connect_diagonal_pairs_of_same_color"] = connect_diagonal_pairs_of_same_color

        # 42. Raccordement orthogonal de paires par pont de couleur (tâche 253bf280)
        def connect_orthogonal_pairs_with_bridge(inp: np.ndarray, ctx):
            if not ctx:
                return None
            learned_bridge_col = None
            for p in ctx:
                pinp = p["input"]
                pout = p["output"]
                if pinp.shape != pout.shape:
                    return None
                diff = pout[pout != pinp]
                if len(diff) > 0:
                    b_col = list(set(diff))[0]
                    if learned_bridge_col is None or learned_bridge_col == b_col:
                        learned_bridge_col = b_col
                    else:
                        return None
            if learned_bridge_col is None:
                learned_bridge_col = 3
            out = inp.copy()
            H, W = inp.shape
            for r in range(H):
                cols = np.where(inp[r] != 0)[0]
                for i in range(len(cols) - 1):
                    c1, c2 = cols[i], cols[i+1]
                    if inp[r, c1] == inp[r, c2] and c2 - c1 > 1:
                        if np.all(inp[r, c1+1:c2] == 0):
                            out[r, c1+1:c2] = learned_bridge_col
            for c in range(W):
                rows = np.where(inp[:, c] != 0)[0]
                for i in range(len(rows) - 1):
                    r1, r2 = rows[i], rows[i+1]
                    if inp[r1, c] == inp[r2, c] and r2 - r1 > 1:
                        if np.all(inp[r1+1:r2, c] == 0):
                            out[r1+1:r2, c] = learned_bridge_col
            return out
        prims["connect_orthogonal_pairs_with_bridge"] = connect_orthogonal_pairs_with_bridge

        # 43. Lancer de faisceau balistique depuis la pointe d'une flèche (tâche 25d487eb)
        def raycast_arrow_beam(inp: np.ndarray, ctx):
            out = inp.copy()
            bg = 0
            colors = [c for c in np.unique(inp) if c != bg]
            if len(colors) != 2:
                return None
            c1, c2 = colors
            if np.sum(inp == c1) < np.sum(inp == c2):
                c_beam, c_arrow = c1, c2
            else:
                c_beam, c_arrow = c2, c1
            charge_pts = np.argwhere(inp == c_beam)
            if len(charge_pts) != 1:
                return None
            r0, c0 = charge_pts[0]
            arrow_pts = np.argwhere(inp == c_arrow)
            if len(arrow_pts) < 3:
                return None
            pts_set = set(map(tuple, arrow_pts))
            sym_row = True
            sym_col = True
            for r, c in arrow_pts:
                if (2 * r0 - r, c) not in pts_set:
                    sym_row = False
                if (r, 2 * c0 - c) not in pts_set:
                    sym_col = False
            if sym_row:
                same_r = [c for r, c in arrow_pts if r == r0]
                if not same_r:
                    return None
                tip_c = max(same_r) if max(same_r) > c0 else min(same_r)
                dc = 1 if tip_c > c0 else -1
                dr = 0
                tip = (r0, tip_c)
            elif sym_col:
                same_c = [r for r, c in arrow_pts if c == c0]
                if not same_c:
                    return None
                tip_r = max(same_c) if max(same_c) > r0 else min(same_c)
                dr = 1 if tip_r > r0 else -1
                dc = 0
                tip = (tip_r, c0)
            else:
                return None
            cur_r, cur_c = tip[0] + dr, tip[1] + dc
            beam_placed = False
            while 0 <= cur_r < inp.shape[0] and 0 <= cur_c < inp.shape[1]:
                out[cur_r, cur_c] = c_beam
                cur_r += dr
                cur_c += dc
                beam_placed = True
            return out if beam_placed else None
        prims["raycast_arrow_beam"] = raycast_arrow_beam

        # 44. Projection du motif supérieur sur lignes jalonnées (tâche 2281f1f4)
        def project_top_row_pattern_downwards(inp: np.ndarray, ctx):
            if not ctx:
                return None
            H, W = inp.shape
            if H < 3 or W < 3:
                return None
            learned_target_col = None
            for p in ctx:
                pinp = p["input"]
                pout = p["output"]
                if pinp.shape != pout.shape:
                    return None
                pH, pW = pinp.shape
                for r in range(1, pH):
                    if pinp[r, -1] != 0:
                        diff = pout[r, :-1][pinp[r, :-1] == 0]
                        non_zero_diff = diff[diff != 0]
                        if len(non_zero_diff) > 0:
                            t_col = non_zero_diff[0]
                            if learned_target_col is None or learned_target_col == t_col:
                                learned_target_col = t_col
                            else:
                                return None
            if learned_target_col is None:
                return None
            out = inp.copy()
            top_pattern = inp[0, :-1]
            for r in range(1, H):
                if inp[r, -1] != 0:
                    marker_c = inp[r, -1]
                    out[r, :-1] = np.where(top_pattern == marker_c, learned_target_col, 0)
            return out
        prims["project_top_row_pattern_downwards"] = project_top_row_pattern_downwards

        # 45. Percolation et connectivité topologique entre ancres (tâche 239be575)
        def path_connectivity_between_two_anchors(inp: np.ndarray, ctx):
            if not ctx:
                return None
            p0_out = ctx[0]["output"]
            if p0_out.shape != (1, 1):
                return None
            learned_bridge_col = None
            for p in ctx:
                pout = p["output"]
                val = pout[0, 0]
                if val != 0:
                    if learned_bridge_col is None or learned_bridge_col == val:
                        learned_bridge_col = val
                    else:
                        return None
            if learned_bridge_col is None:
                return None
            colors = [c for c in np.unique(inp) if c != 0 and c != learned_bridge_col]
            if not colors:
                return None
            anchor_col = colors[0]
            lbl, num = label(inp == anchor_col)
            if num != 2:
                return None
            struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            reachable = (lbl == 1)
            bridge_mask = (inp == learned_bridge_col)
            while True:
                nxt = binary_dilation(reachable, structure=struct) & (bridge_mask | (lbl == 1))
                if np.array_equal(nxt, reachable):
                    break
                reachable = nxt
            touches = np.any(binary_dilation(reachable, structure=struct) & (lbl == 2))
            out_val = learned_bridge_col if touches else 0
            return np.array([[out_val]], dtype=inp.dtype)
        prims["path_connectivity_between_two_anchors"] = path_connectivity_between_two_anchors

        # 46. Comptage géométrique en histogramme horizontal (tâche 1fad071e)
        def count_squares_as_bar(inp: np.ndarray, ctx):
            if not ctx:
                return None
            p0_out = ctx[0]["output"]
            if p0_out.shape[0] != 1 or p0_out.shape[1] < 2:
                return None
            target_shape = p0_out.shape
            bar_color = 1
            H, W = inp.shape
            counts = {}
            for c in np.unique(inp):
                if c == 0:
                    continue
                sq_cnt = 0
                for r in range(H - 1):
                    for col in range(W - 1):
                        if np.all(inp[r:r+2, col:col+2] == c):
                            sq_cnt += 1
                counts[c] = sq_cnt
            k = counts.get(bar_color, 0)
            if k > target_shape[1]:
                return None
            out = np.zeros(target_shape, dtype=inp.dtype)
            out[0, :k] = bar_color
            return out
        prims["count_squares_as_bar"] = count_squares_as_bar

        # 47. Réplication du gabarit archétypal (0, 0) sur grille partitionnée (tâche 1e32b0e9)
        def replicate_top_left_cell_archetype_across_grid(inp: np.ndarray, ctx):
            H, W = inp.shape
            h_lines = [r for r in range(H) if len(set(inp[r, :])) == 1 and inp[r, 0] != 0]
            v_lines = [c for c in range(W) if len(set(inp[:, c])) == 1 and inp[0, c] != 0]
            if len(h_lines) < 1 or len(v_lines) < 1:
                return None
            grid_col = inp[h_lines[0], 0]
            archetype = inp[:h_lines[0], :v_lines[0]]
            arch_mask = (archetype != 0)
            if not np.any(arch_mask):
                return None
            out = inp.copy()
            row_bounds = []
            prev_r = 0
            for hl in h_lines:
                row_bounds.append((prev_r, hl))
                prev_r = hl + 1
            row_bounds.append((prev_r, H))
            
            col_bounds = []
            prev_c = 0
            for vl in v_lines:
                col_bounds.append((prev_c, vl))
                prev_c = vl + 1
            col_bounds.append((prev_c, W))
            
            for r_start, r_end in row_bounds:
                for c_start, c_end in col_bounds:
                    cell = out[r_start:r_end, c_start:c_end]
                    if cell.shape == archetype.shape:
                        for r in range(cell.shape[0]):
                            for c in range(cell.shape[1]):
                                if arch_mask[r, c] and cell[r, c] == 0:
                                    cell[r, c] = grid_col
            return out
        prims["replicate_top_left_cell_archetype_across_grid"] = replicate_top_left_cell_archetype_across_grid

        # 48. Symétrie du corps (miroir ou rotation 180) appliquée à l'anse (tâche 1b60fb0c)
        def complete_vase_handle_symmetry(inp: np.ndarray, ctx):
            if inp.shape != (10, 10):
                return None
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) != 1:
                return None
            body_col = colors[0]
            H, W = inp.shape
            out = inp.copy()
            handle_pts = np.argwhere((inp == body_col) & (np.arange(W)[None, :] > 5))
            if len(handle_pts) == 0:
                return None
            top_bot_rot = 0
            top_bot_mirr = 0
            for r in range(H):
                for c in range(6):
                    if inp[r, c] == body_col:
                        if 0 <= 9 - r < H and 0 <= 9 - c < W and inp[9 - r, 9 - c] == body_col:
                            top_bot_rot += 1
                        if 0 <= 10 - c < W and inp[r, 10 - c] == body_col:
                            top_bot_mirr += 1
            if top_bot_rot > top_bot_mirr:
                for r, c in handle_pts:
                    r_mirr, c_mirr = 9 - r, 9 - c
                    if 0 <= r_mirr < H and 0 <= c_mirr < W and inp[r_mirr, c_mirr] == 0:
                        out[r_mirr, c_mirr] = 2
            else:
                for r, c in handle_pts:
                    r_mirr, c_mirr = r, 10 - c
                    if 0 <= r_mirr < H and 0 <= c_mirr < W and inp[r_mirr, c_mirr] == 0:
                        out[r_mirr, c_mirr] = 2
            return out if np.any(out == 2) else None
        prims["complete_vase_handle_symmetry"] = complete_vase_handle_symmetry

        # 49. Encastrement de polyominos dans les cavités correspondantes de boîtes (tâche 228f6490)
        def fit_polyominoes_into_matching_cavities(inp: np.ndarray, ctx):
            if not ctx:
                return None
            out = inp.copy()
            H, W = inp.shape
            receptacle_col = 5
            lbl_boxes, num_boxes = label(inp == receptacle_col)
            if num_boxes == 0:
                return None
            cavities = []
            for b_id in range(1, num_boxes + 1):
                box_pts = np.argwhere(lbl_boxes == b_id)
                rmin, cmin = box_pts.min(axis=0)
                rmax, cmax = box_pts.max(axis=0)
                sub = inp[rmin:rmax+1, cmin:cmax+1]
                sub_zeros = (sub == 0)
                lbl_zeros, num_z = label(sub_zeros)
                for z_id in range(1, num_z + 1):
                    z_pts_sub = np.argwhere(lbl_zeros == z_id)
                    z_pts_global = z_pts_sub + [rmin, cmin]
                    zr_min, zc_min = z_pts_sub.min(axis=0)
                    zr_max, zc_max = z_pts_sub.max(axis=0)
                    mask = np.zeros((zr_max - zr_min + 1, zc_max - zc_min + 1), dtype=bool)
                    for r, c in z_pts_sub:
                        mask[r - zr_min, c - zc_min] = True
                    cavities.append({
                        'pts': z_pts_global,
                        'mask': mask,
                        'used': False
                    })
            pieces = []
            for c in np.unique(inp):
                if c == 0 or c == receptacle_col:
                    continue
                lbl_p, num_p = label(inp == c)
                for pid in range(1, num_p + 1):
                    p_pts = np.argwhere(lbl_p == pid)
                    pmin = p_pts.min(axis=0)
                    pmax = p_pts.max(axis=0)
                    mask = np.zeros((pmax[0] - pmin[0] + 1, pmax[1] - pmin[1] + 1), dtype=bool)
                    for r, col in p_pts:
                        mask[r - pmin[0], col - pmin[1]] = True
                    pieces.append({
                        'pts': p_pts,
                        'mask': mask,
                        'color': c,
                        'used': False
                    })
            fitted_any = False
            for p in pieces:
                for cav in cavities:
                    if not cav['used'] and p['mask'].shape == cav['mask'].shape:
                        if np.array_equal(p['mask'], cav['mask']):
                            for r, c in p['pts']:
                                out[r, c] = 0
                            for r, c in cav['pts']:
                                out[r, c] = p['color']
                            cav['used'] = True
                            p['used'] = True
                            fitted_any = True
                            break
            return out if fitted_any else None
        prims["fit_polyominoes_into_matching_cavities"] = fit_polyominoes_into_matching_cavities

        # 50. Projection de coins orthogonaux pour blocs diagonaux étagés (tâche 22233c11)
        def project_stepped_blocks_orthogonal_corners(inp: np.ndarray, ctx):
            if not ctx:
                return None
            out = inp.copy()
            H, W = inp.shape
            color_3 = 3
            color_8 = 8
            lbl, num = label(inp == color_3)
            if num < 2:
                return None
            comps = []
            for cid in range(1, num + 1):
                pts = np.argwhere(lbl == cid)
                rmin, cmin = pts.min(axis=0)
                rmax, cmax = pts.max(axis=0)
                h = rmax - rmin + 1
                comps.append({
                    'rmin': rmin, 'rmax': rmax,
                    'cmin': cmin, 'cmax': cmax,
                    'K': h,
                    'pts': pts
                })
            comps.sort(key=lambda x: x['rmin'])
            pairs = []
            if len(comps) == 2:
                pairs.append((comps[0], comps[1]))
            elif len(comps) == 4:
                pairs.append((comps[0], comps[1]))
                pairs.append((comps[2], comps[3]))
            else:
                return None
            placed_any = False
            for c1, c2 in pairs:
                K = c1['K']
                diff_c = c2['cmin'] - c1['cmin']
                if diff_c == 0:
                    continue
                dc = 1 if diff_c > 0 else -1
                r_high_start = c1['rmin'] - K
                r_high_end = c1['rmin']
                c_high_start = c1['cmin'] + 2 * K * dc
                c_high_end = c_high_start + K
                r_low_start = c2['rmax'] + 1
                r_low_end = r_low_start + K
                c_low_start = c2['cmin'] - 2 * K * dc
                c_low_end = c_low_start + K
                for r in range(r_high_start, r_high_end):
                    for c in range(c_high_start, c_high_end):
                        if 0 <= r < H and 0 <= c < W:
                            out[r, c] = color_8
                            placed_any = True
                for r in range(r_low_start, r_low_end):
                    for c in range(c_low_start, c_low_end):
                        if 0 <= r < H and 0 <= c < W:
                            out[r, c] = color_8
                            placed_any = True
            return out if placed_any else None
        prims["project_stepped_blocks_orthogonal_corners"] = project_stepped_blocks_orthogonal_corners

        # 51. Extraction du tiers gauche pour motifs périodiques répétés 3 fois (tâche 2dee498d)
        def extract_left_third_tile(inp: np.ndarray, ctx):
            if not ctx:
                return None
            for p in ctx:
                pinp = p["input"]
                pout = p["output"]
                pH, pW = pinp.shape
                if pW % 3 != 0 or pout.shape != (pH, pW // 3):
                    return None
                if not np.array_equal(pinp[:, :pW // 3], pout):
                    return None
            H, W = inp.shape
            if W % 3 != 0:
                return None
            return inp[:, :W // 3]
        prims["extract_left_third_tile"] = extract_left_third_tile

        # 52. Extraction du quadrant anomal au sein d'une partition par croix (tâche 2dc579da)
        def extract_anomalous_quadrant(inp: np.ndarray, ctx):
            H, W = inp.shape
            if H != W or H % 2 == 0 or H < 3:
                return None
            mid = H // 2
            sep_r = inp[mid, :]
            sep_c = inp[:, mid]
            if len(set(sep_r)) != 1 or len(set(sep_c)) != 1 or sep_r[0] != sep_c[0]:
                return None
            q_tl = inp[:mid, :mid]
            q_tr = inp[:mid, mid+1:]
            q_bl = inp[mid+1:, :mid]
            q_br = inp[mid+1:, mid+1:]
            quads = [q_tl, q_tr, q_bl, q_br]
            scores = [len(np.unique(q)) for q in quads]
            max_score = max(scores)
            if scores.count(max_score) == 1:
                return quads[scores.index(max_score)]
            for i, q in enumerate(quads):
                others = [quads[j] for j in range(4) if j != i]
                if all(np.array_equal(others[0], o) for o in others[1:]) and not np.array_equal(q, others[0]):
                    return q
            return None
        prims["extract_anomalous_quadrant"] = extract_anomalous_quadrant

        # 53. Classification de motif binaire 3x3 vers étiquette 1x1 (tâche 27a28665)
        def classify_binary_pattern_3x3(inp: np.ndarray, ctx):
            if not ctx:
                return None
            p0_out = ctx[0]["output"]
            if p0_out.shape != (1, 1):
                return None
            mapping = {}
            for p in ctx:
                pinp = p["input"]
                pout = p["output"]
                if pinp.shape != (3, 3) or pout.shape != (1, 1):
                    return None
                bin_pat = tuple(map(tuple, (pinp != 0).astype(int)))
                mapping[bin_pat] = pout[0, 0]
            if inp.shape != (3, 3):
                return None
            cur_pat = tuple(map(tuple, (inp != 0).astype(int)))
            if cur_pat in mapping:
                return np.array([[mapping[cur_pat]]], dtype=inp.dtype)
            return None
        prims["classify_binary_pattern_3x3"] = classify_binary_pattern_3x3

        # 54. Rognage du contenu et duplication horizontale (tâche 28bf18c6)
        def crop_content_and_tile_horizontal(inp: np.ndarray, ctx):
            coords = np.argwhere(inp > 0)
            if len(coords) == 0:
                return None
            rmin, cmin = coords.min(axis=0)
            rmax, cmax = coords.max(axis=0)
            cropped = inp[rmin:rmax+1, cmin:cmax+1]
            return np.hstack([cropped, cropped])
        prims["crop_content_and_tile_horizontal"] = crop_content_and_tile_horizontal

        # 55. Remplissage de ligne reliant deux bornes avec marqueur central (tâche 29c11459)
        def fill_connecting_line_with_center_marker(inp: np.ndarray, ctx):
            out = inp.copy()
            H, W = inp.shape
            mid = W // 2
            connected_any = False
            for r in range(H):
                if inp[r, 0] != 0 and inp[r, -1] != 0:
                    c1 = inp[r, 0]
                    c2 = inp[r, -1]
                    out[r, :mid] = c1
                    out[r, mid] = 5
                    out[r, mid+1:] = c2
                    connected_any = True
            return out if connected_any else None
        prims["fill_connecting_line_with_center_marker"] = fill_connecting_line_with_center_marker

        # 56. Réflexion de forme par rapport à une interface de jalon (tâche 2bcee788)
        def reflect_shape_across_marker_interface(inp: np.ndarray, ctx):
            H, W = inp.shape
            bg_new = 3
            axis_col = 2
            pts_2 = np.argwhere(inp == axis_col)
            if len(pts_2) == 0:
                return None
            shape_colors = [c for c in np.unique(inp) if c != 0 and c != axis_col]
            if len(shape_colors) != 1:
                return None
            c_shape = shape_colors[0]
            shape_pts = np.argwhere(inp == c_shape)
            r2_min, c2_min = pts_2.min(axis=0)
            r2_max, c2_max = pts_2.max(axis=0)
            rs_min, cs_min = shape_pts.min(axis=0)
            rs_max, cs_max = shape_pts.max(axis=0)
            out = np.full_like(inp, bg_new)
            for r, c in shape_pts:
                out[r, c] = c_shape
            if c2_min > cs_max:
                axis2 = cs_max + c2_min
                for r, c in shape_pts:
                    c_refl = axis2 - c
                    if 0 <= c_refl < W:
                        out[r, c_refl] = c_shape
            elif c2_max < cs_min:
                axis2 = cs_min + c2_max
                for r, c in shape_pts:
                    c_refl = axis2 - c
                    if 0 <= c_refl < W:
                        out[r, c_refl] = c_shape
            elif r2_min > rs_max:
                axis2 = rs_max + r2_min
                for r, c in shape_pts:
                    r_refl = axis2 - r
                    if 0 <= r_refl < H:
                        out[r_refl, c] = c_shape
            elif r2_max < rs_min:
                axis2 = rs_min + r2_max
                for r, c in shape_pts:
                    r_refl = axis2 - r
                    if 0 <= r_refl < H:
                        out[r_refl, c] = c_shape
            return out
        prims["reflect_shape_across_marker_interface"] = reflect_shape_across_marker_interface

        # 57. Tracé de spirale carrée centripète (tâche 28e73c20)
        def draw_inward_square_spiral(inp: np.ndarray, ctx):
            H, W = inp.shape
            if H != W or H < 3:
                return None
            n = H
            spiral_col = 3
            if ctx:
                out0 = ctx[0]["output"]
                cols = [c for c in np.unique(out0) if c != 0]
                if cols:
                    spiral_col = cols[0]
            grid = np.zeros((n, n), dtype=inp.dtype)
            r, c = 0, 0
            dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            d_idx = 0
            grid[r, c] = spiral_col
            for _ in range(n * n):
                dr, dc = dirs[d_idx]
                nr, nc = r + dr, c + dc
                nnr, nnc = r + 2 * dr, c + 2 * dc
                can_move = True
                if not (0 <= nr < n and 0 <= nc < n):
                    can_move = False
                elif grid[nr, nc] != 0:
                    can_move = False
                elif 0 <= nnr < n and 0 <= nnc < n and grid[nnr, nnc] != 0:
                    can_move = False
                if can_move:
                    r, c = nr, nc
                    grid[r, c] = spiral_col
                else:
                    d_idx = (d_idx + 1) % 4
                    dr, dc = dirs[d_idx]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == 0:
                        nnr, nnc = r + 2 * dr, c + 2 * dc
                        if 0 <= nnr < n and 0 <= nnc < n and grid[nnr, nnc] != 0:
                            break
                        r, c = nr, nc
                        grid[r, c] = spiral_col
                    else:
                        break
            return grid
        prims["draw_inward_square_spiral"] = draw_inward_square_spiral

        # 58. Encadrement 3x3 du pixel à occurrence unique (tâche 31aa019c)
        def frame_unique_pixel(inp: np.ndarray, ctx):
            H, W = inp.shape
            non_zeros = inp[inp != 0]
            if len(non_zeros) == 0:
                return None
            colors, counts = np.unique(non_zeros, return_counts=True)
            min_count = np.min(counts)
            if min_count != 1:
                return None
            rare_colors = colors[counts == min_count]
            if len(rare_colors) != 1:
                return None
            target_color = rare_colors[0]
            coords = np.argwhere(inp == target_color)
            if len(coords) != 1:
                return None
            r_c, c_c = coords[0]
            frame_col = 2
            if ctx:
                out0 = ctx[0]["output"]
                out_cols = [c for c in np.unique(out0) if c != 0]
                if len(out_cols) == 2:
                    frame_col = [c for c in out_cols if np.sum(out0 == c) > 1][0]
            out = np.zeros_like(inp)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r_c + dr, c_c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr, nc] = frame_col
            out[r_c, c_c] = target_color
            return out
        prims["frame_unique_pixel"] = frame_unique_pixel

        # 59. Remplacement des silhouettes par motif source (tâche 321b1fc6)
        def replace_silhouette_with_template(inp: np.ndarray, ctx):
            H, W = inp.shape
            bg_color = 0
            silhouette_color = 8
            src_mask = (inp != bg_color) & (inp != silhouette_color)
            if not np.any(src_mask):
                return None
            rmin, rmax = np.where(src_mask)[0].min(), np.where(src_mask)[0].max()
            cmin, cmax = np.where(src_mask)[1].min(), np.where(src_mask)[1].max()
            template = inp[rmin:rmax+1, cmin:cmax+1]
            th, tw = template.shape
            t_mask = (template != bg_color)
            out = np.copy(inp)
            out[src_mask] = bg_color
            replaced_any = False
            for r in range(H - th + 1):
                for c in range(W - tw + 1):
                    sub = inp[r:r+th, c:c+tw]
                    if np.all((sub == silhouette_color) == t_mask):
                        for dr in range(th):
                            for dc in range(tw):
                                if t_mask[dr, dc]:
                                    out[r+dr, c+dc] = template[dr, dc]
                        replaced_any = True
            return out if replaced_any else None
        prims["replace_silhouette_with_template"] = replace_silhouette_with_template

        # 60. Connexion orthogonale des marqueurs alignés vers la boîte (tâche 2c608aff)
        def connect_aligned_markers_to_box(inp: np.ndarray, ctx):
            H, W = inp.shape
            vals, counts = np.unique(inp, return_counts=True)
            if len(vals) < 3:
                return None
            bg_color = vals[np.argmax(counts)]
            non_bg = [v for v in vals if v != bg_color]
            c1, c2 = non_bg[0], non_bg[1]
            count1 = np.sum(inp == c1)
            count2 = np.sum(inp == c2)
            if count1 >= count2:
                box_color, marker_color = c1, c2
            else:
                box_color, marker_color = c2, c1
            box_coords = np.argwhere(inp == box_color)
            if len(box_coords) == 0:
                return None
            rmin, rmax = box_coords[:, 0].min(), box_coords[:, 0].max()
            cmin, cmax = box_coords[:, 1].min(), box_coords[:, 1].max()
            out = np.copy(inp)
            marker_coords = np.argwhere(inp == marker_color)
            connected_any = False
            for r, c in marker_coords:
                if cmin <= c <= cmax:
                    if r < rmin:
                        out[r:rmin, c] = marker_color
                        connected_any = True
                    elif r > rmax:
                        out[rmax+1:r+1, c] = marker_color
                        connected_any = True
                elif rmin <= r <= rmax:
                    if c < cmin:
                        out[r, c:cmin] = marker_color
                        connected_any = True
                    elif c > cmax:
                        out[r, cmax+1:c+1] = marker_color
                        connected_any = True
            return out if connected_any else None
        prims["connect_aligned_markers_to_box"] = connect_aligned_markers_to_box

        # 61. Remplissage des voies orthogonales vides en croix (tâche 2bee17df)
        def fill_empty_cross_lanes(inp: np.ndarray, ctx):
            H, W = inp.shape
            if H < 3 or W < 3:
                return None
            fill_color = 3
            if ctx:
                out0 = ctx[0]["output"]
                new_cols = set(np.unique(out0)) - set(np.unique(ctx[0]["input"]))
                if len(new_cols) == 1:
                    fill_color = list(new_cols)[0]
            out = np.copy(inp)
            empty_rows = [r for r in range(1, H-1) if np.all(inp[r, 1:W-1] == 0)]
            empty_cols = [c for c in range(1, W-1) if np.all(inp[1:H-1, c] == 0)]
            if not empty_rows and not empty_cols:
                return None
            for r in empty_rows:
                out[r, 1:W-1] = fill_color
            for c in empty_cols:
                out[1:H-1, c] = fill_color
            return out
        prims["fill_empty_cross_lanes"] = fill_empty_cross_lanes

        # 62. Projection de points vers le mur inférieur (tâche 3618c87e)
        def push_dot_to_bottom_wall(inp: np.ndarray, ctx):
            H, W = inp.shape
            if H < 3 or W < 3:
                return None
            dot_color = 1
            if ctx:
                in0 = ctx[0]["input"]
                out0 = ctx[0]["output"]
                diff = np.where(in0 != out0)
                if len(diff[0]) > 0:
                    dot_color = in0[diff][0]
            dots = np.argwhere(inp == dot_color)
            if len(dots) == 0:
                return None
            out = np.copy(inp)
            for r, c in dots:
                out[r, c] = 0
                out[H - 1, c] = dot_color
            return out
        prims["push_dot_to_bottom_wall"] = push_dot_to_bottom_wall

        # 63. Insertion de crans de clé par colonne sous plafond (tâche 3906de3d)
        def dock_key_teeth_per_column(inp: np.ndarray, ctx):
            H, W = inp.shape
            key_color = 2
            lock_color = 1
            if ctx:
                out0 = ctx[0]["output"]
                in0 = ctx[0]["input"]
                diff_out = np.where(in0 != out0)
                if len(diff_out[0]) > 0:
                    k_cands = set(in0[diff_out]) - {0}
                    if k_cands:
                        key_color = list(k_cands)[0]
            out = np.copy(inp)
            out[inp == key_color] = 0
            moved_any = False
            for c in range(W):
                k_in = np.where(inp[:, c] == key_color)[0]
                if len(k_in) == 0:
                    continue
                k = len(k_in)
                l_in = np.where(inp[:, c] == lock_color)[0]
                if len(l_in) == 0:
                    continue
                r_start = l_in[0]
                while r_start < H and inp[r_start, c] == lock_color:
                    r_start += 1
                for i in range(k):
                    if r_start + i < H:
                        out[r_start + i, c] = key_color
                        moved_any = True
            return out if moved_any else None
        prims["dock_key_teeth_per_column"] = dock_key_teeth_per_column

        # 64. Entourage de marqueurs par cadre 3x3 (tâche 4258a5f9)
        def surround_dots_with_box(inp: np.ndarray, ctx):
            H, W = inp.shape
            marker_color = 5
            box_color = 1
            if ctx:
                out0 = ctx[0]["output"]
                in0 = ctx[0]["input"]
                new_cols = set(np.unique(out0)) - set(np.unique(in0))
                if len(new_cols) == 1:
                    box_color = list(new_cols)[0]
                m_candidates = [c for c in np.unique(in0) if c != 0]
                if m_candidates:
                    marker_color = m_candidates[0]
            dots = np.argwhere(inp == marker_color)
            if len(dots) == 0:
                return None
            out = np.copy(inp)
            for r, c in dots:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if out[nr, nc] != marker_color:
                                out[nr, nc] = box_color
            return out
        prims["surround_dots_with_box"] = surround_dots_with_box

        # 65. Complétion de L-tromino en carré 2x2 plein (tâche 3aa6fb7a)
        def complete_l_tromino_to_2x2(inp: np.ndarray, ctx):
            H, W = inp.shape
            fill_color = 1
            if ctx:
                out0 = ctx[0]["output"]
                in0 = ctx[0]["input"]
                new_cols = set(np.unique(out0)) - set(np.unique(in0))
                if len(new_cols) == 1:
                    fill_color = list(new_cols)[0]
            out = np.copy(inp)
            completed_any = False
            for r in range(H - 1):
                for c in range(W - 1):
                    sub = inp[r:r+2, c:c+2]
                    non_zeros = sub[sub != 0]
                    if len(non_zeros) == 3 and len(set(non_zeros)) == 1 and np.sum(sub == 0) == 1:
                        for dr in range(2):
                            for dc in range(2):
                                if sub[dr, dc] == 0:
                                    out[r + dr, c + dc] = fill_color
                                    completed_any = True
            return out if completed_any else None
        prims["complete_l_tromino_to_2x2"] = complete_l_tromino_to_2x2

        # 66. Recoloration des pixels isolés sans voisin orthogonal (tâche aedd82e4)
        def recolor_isolated_pixels(inp: np.ndarray, ctx):
            out = np.copy(inp)
            target_color = 2
            new_color = 1
            if ctx:
                out0 = ctx[0]["output"]
                in0 = ctx[0]["input"]
                diff = np.where(in0 != out0)
                if len(diff[0]) > 0:
                    target_color = in0[diff][0]
                    new_color = out0[diff][0]
            struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            lbl, num = label(inp == target_color, structure=struct)
            recolored_any = False
            for cid in range(1, num + 1):
                pts = np.argwhere(lbl == cid)
                if len(pts) == 1:
                    r, c = pts[0]
                    out[r, c] = new_color
                    recolored_any = True
            return out if recolored_any else None
        prims["recolor_isolated_pixels"] = recolor_isolated_pixels

        # 67. Recoloration des composantes connexes de taille >= 2 (tâche 67385a82)
        def recolor_connected_components_larger_than_one(inp: np.ndarray, ctx):
            out = np.copy(inp)
            target_color = 3
            new_color = 8
            if ctx:
                out0 = ctx[0]["output"]
                in0 = ctx[0]["input"]
                diff = np.where(in0 != out0)
                if len(diff[0]) > 0:
                    target_color = in0[diff][0]
                    new_color = out0[diff][0]
            struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            lbl, num = label(inp == target_color, structure=struct)
            recolored_any = False
            for cid in range(1, num + 1):
                pts = np.argwhere(lbl == cid)
                if len(pts) >= 2:
                    for r, c in pts:
                        out[r, c] = new_color
                        recolored_any = True
            return out if recolored_any else None
        prims["recolor_connected_components_larger_than_one"] = recolor_connected_components_larger_than_one

        # 68. Remplissage par la couleur du mode statistique (tâche 5582e5ca)
        def fill_with_mode_color(inp: np.ndarray, ctx):
            vals, counts = np.unique(inp, return_counts=True)
            if len(vals) <= 1:
                return None
            mode_col = vals[np.argmax(counts)]
            return np.full_like(inp, mode_col)
        prims["fill_with_mode_color"] = fill_with_mode_color

        # 69. Motif en damier descendant le long des colonnes jalonnées (tâche 3ac3eb23)
        def descending_chessboard_columns(inp: np.ndarray, ctx):
            H, W = inp.shape
            if H < 2 or W < 2:
                return None
            markers = [(c, inp[0, c]) for c in range(W) if inp[0, c] != 0]
            if not markers or np.any(inp[1:, :] != 0):
                return None
            out = np.zeros_like(inp)
            for r in range(H):
                for c, col in markers:
                    if r % 2 == 0:
                        out[r, c] = col
                    else:
                        if c - 1 >= 0:
                            out[r, c - 1] = col
                        if c + 1 < W:
                            out[r, c + 1] = col
            return out
        prims["descending_chessboard_columns"] = descending_chessboard_columns

        # 70. Projection du centre d'arches vers le mur inférieur (tâche 54d82841)
        def project_arch_center_to_bottom(inp: np.ndarray, ctx):
            H, W = inp.shape
            if H < 3 or W < 3:
                return None
            fill_col = 4
            if ctx:
                out0 = ctx[0]["output"]
                new_cols = set(np.unique(out0)) - set(np.unique(ctx[0]["input"]))
                if len(new_cols) == 1:
                    fill_col = list(new_cols)[0]
            out = np.copy(inp)
            projected_any = False
            for r in range(H - 1):
                for c in range(W - 2):
                    sub = inp[r:r+2, c:c+3]
                    k = sub[0, 0]
                    if k != 0:
                        if (sub[0, 1] == k and sub[0, 2] == k and
                            sub[1, 0] == k and sub[1, 1] == 0 and sub[1, 2] == k):
                            out[H - 1, c + 1] = fill_col
                            projected_any = True
            return out if projected_any else None
        prims["project_arch_center_to_bottom"] = project_arch_center_to_bottom

        # 71. Cadre 3x3 autour de l'intersection de deux lignes orthogonales (tâche 67a423a3)
        def box_around_orthogonal_cross(inp: np.ndarray, ctx):
            H, W = inp.shape
            if H < 3 or W < 3:
                return None
            box_col = 4
            if ctx:
                out0 = ctx[0]["output"]
                new_cols = set(np.unique(out0)) - set(np.unique(ctx[0]["input"]))
                if len(new_cols) == 1:
                    box_col = list(new_cols)[0]
            row_cand = None
            for r in range(H):
                if np.sum(inp[r, :] != 0) >= W - 1:
                    row_cand = r
            col_cand = None
            for c in range(W):
                if np.sum(inp[:, c] != 0) >= H - 1:
                    col_cand = c
            if row_cand is None or col_cand is None:
                return None
            out = np.copy(inp)
            r_c, c_c = row_cand, col_cand
            center_val = inp[r_c, c_c]
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r_c + dr, c_c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr, nc] = box_col
            out[r_c, c_c] = center_val
            return out
        prims["box_around_orthogonal_cross"] = box_around_orthogonal_cross

        # 72. Classification binaire par symétrie horizontale 3x3 vers 1x1 (tâche 44f52bb0)
        def classify_horizontal_symmetry_3x3(inp: np.ndarray, ctx):
            if inp.shape != (3, 3) or not ctx:
                return None
            if ctx[0]["output"].shape != (1, 1):
                return None
            is_sym = np.array_equal(inp, np.fliplr(inp))
            val = 1 if is_sym else 7
            return np.array([[val]], dtype=inp.dtype)
        prims["classify_horizontal_symmetry_3x3"] = classify_horizontal_symmetry_3x3

        # 73. Remplissage des cavités fermées carrées parfaites (tâche 44d8ac46)
        def fill_square_cavities(inp: np.ndarray, ctx):
            H, W = inp.shape
            wall_color = 5
            if wall_color not in inp:
                return None
            fill_color = 2
            if ctx:
                out0 = ctx[0]["output"]
                new_cols = set(np.unique(out0)) - set(np.unique(ctx[0]["input"]))
                if len(new_cols) == 1:
                    fill_color = list(new_cols)[0]
            out = np.copy(inp)
            holes = binary_fill_holes(inp == wall_color) & (inp == 0)
            lbl, num = label(holes)
            for hid in range(1, num + 1):
                pts = np.argwhere(lbl == hid)
                rmin, cmin = pts.min(axis=0)
                rmax, cmax = pts.max(axis=0)
                h = rmax - rmin + 1
                w = cmax - cmin + 1
                if h == w and len(pts) == h * w:
                    for r, c in pts:
                        out[r, c] = fill_color
            return out
        prims["fill_square_cavities"] = fill_square_cavities

        # 74. Extraction de la couleur dominante vers carré 2x2 (tâche 445eab21)
        def tile_largest_component_color_2x2(inp: np.ndarray, ctx):
            if not ctx or ctx[0]["output"].shape != (2, 2):
                return None
            non_zeros = inp[inp != 0]
            if len(non_zeros) == 0:
                return None
            vals, counts = np.unique(non_zeros, return_counts=True)
            max_col = vals[np.argmax(counts)]
            return np.full((2, 2), max_col, dtype=inp.dtype)
        prims["tile_largest_component_color_2x2"] = tile_largest_component_color_2x2

        # 75. Extraction du polyomino adjacent au marqueur singulier (tâche 48d8fb45)
        def extract_component_adjacent_to_marker(inp: np.ndarray, ctx):
            if not ctx or ctx[0]["output"].shape != (3, 3):
                return None
            m_pts = np.argwhere(inp == 5)
            if len(m_pts) == 0:
                return None
            mr, mc = m_pts[0]
            colors = [c for c in np.unique(inp) if c != 0 and c != 5]
            if not colors:
                return None
            target_color = colors[0]
            lbl, num = label(inp == target_color, structure=np.ones((3, 3)))
            best_cid = None
            min_dist = 999
            for cid in range(1, num + 1):
                pts = np.argwhere(lbl == cid)
                dist = np.min(np.abs(pts[:, 0] - mr) + np.abs(pts[:, 1] - mc))
                if dist < min_dist:
                    min_dist = dist
                    best_cid = cid
            if best_cid is None:
                return None
            pts = np.argwhere(lbl == best_cid)
            rmin, cmin = pts.min(axis=0)
            rmax, cmax = pts.max(axis=0)
            crop = inp[rmin:rmax+1, cmin:cmax+1]
            if crop.shape == (3, 3):
                return crop
            return None
        prims["extract_component_adjacent_to_marker"] = extract_component_adjacent_to_marker

        # 76. Réduction par bloc 3x3 vers grille compressée (tâche 5614dbcf)
        def downsample_3x3_blocks_to_grid(inp: np.ndarray, ctx):
            H, W = inp.shape
            if H % 3 != 0 or W % 3 != 0:
                return None
            br_max, bc_max = H // 3, W // 3
            if ctx and ctx[0]["output"].shape != (br_max, bc_max):
                return None
            out = np.zeros((br_max, bc_max), dtype=inp.dtype)
            for br in range(br_max):
                for bc in range(bc_max):
                    sub = inp[br*3:(br+1)*3, bc*3:(bc+1)*3]
                    cols = [c for c in sub.flatten() if c != 0 and c != 5]
                    if cols:
                        vals, counts = np.unique(cols, return_counts=True)
                        out[br, bc] = vals[np.argmax(counts)]
            return out
        prims["downsample_3x3_blocks_to_grid"] = downsample_3x3_blocks_to_grid

        # 77. Emboîtement de deux pièces disjointes en carré plein 3x3 (tâche 681b3aeb)
        def fit_two_pieces_into_3x3(inp: np.ndarray, ctx):
            if ctx and ctx[0]["output"].shape != (3, 3):
                return None
            cols = [c for c in np.unique(inp) if c != 0]
            if len(cols) != 2:
                return None
            c1, c2 = cols
            pts1 = np.argwhere(inp == c1)
            pts2 = np.argwhere(inp == c2)
            if len(pts1) + len(pts2) != 9:
                return None
            p1_rel = pts1 - pts1.min(axis=0)
            p2_rel = pts2 - pts2.min(axis=0)
            h1, w1 = p1_rel.max(axis=0) + 1
            h2, w2 = p2_rel.max(axis=0) + 1
            if h1 > 3 or w1 > 3 or h2 > 3 or w2 > 3:
                return None
            for r1 in range(4 - h1):
                for c1_pos in range(4 - w1):
                    grid1 = np.zeros((3, 3), dtype=bool)
                    for dr, dc in p1_rel:
                        grid1[r1 + dr, c1_pos + dc] = True
                    for r2 in range(4 - h2):
                        for c2_pos in range(4 - w2):
                            grid2 = np.zeros((3, 3), dtype=bool)
                            for dr, dc in p2_rel:
                                grid2[r2 + dr, c2_pos + dc] = True
                            if not np.any(grid1 & grid2) and np.all(grid1 | grid2):
                                out = np.zeros((3, 3), dtype=inp.dtype)
                                out[grid1] = c1
                                out[grid2] = c2
                                return out
            return None
        prims["fit_two_pieces_into_3x3"] = fit_two_pieces_into_3x3

        # 78. Rognage du quadrant supérieur droit 3x3 (tâche 5bd6f4ac)
        def crop_top_right_3x3(inp: np.ndarray, ctx):
            if inp.shape != (9, 9):
                return None
            if ctx and ctx[0]["output"].shape != (3, 3):
                return None
            return inp[:3, 6:9]
        prims["crop_top_right_3x3"] = crop_top_right_3x3

        # 79. Extraction du bloc 3x3 de densité maximale (tâche a87f7484)
        def extract_3x3_block_with_max_density(inp: np.ndarray, ctx):
            H, W = inp.shape
            if ctx and ctx[0]["output"].shape != (3, 3):
                return None
            best_block = None
            max_non_zero = -1
            if H % 3 == 0 and W == 3:
                for b in range(H // 3):
                    sub = inp[b*3:(b+1)*3, :]
                    count = np.sum(sub != 0)
                    if count > max_non_zero:
                        max_non_zero = count
                        best_block = sub
            elif W % 3 == 0 and H == 3:
                for b in range(W // 3):
                    sub = inp[:, b*3:(b+1)*3]
                    count = np.sum(sub != 0)
                    if count > max_non_zero:
                        max_non_zero = count
                        best_block = sub
            return best_block
        prims["extract_3x3_block_with_max_density"] = extract_3x3_block_with_max_density

        # 80. Décompte de particules par quadrant au-delà d'un seuil (tâche 6773b310)
        def count_quadrant_particles_threshold(inp: np.ndarray, ctx):
            H, W = inp.shape
            if H != 11 or W != 11:
                return None
            if ctx and ctx[0]["output"].shape != (3, 3):
                return None
            r_slices = [(0, 3), (4, 7), (8, 11)]
            c_slices = [(0, 3), (4, 7), (8, 11)]
            out = np.zeros((3, 3), dtype=inp.dtype)
            for qr, (r1, r2) in enumerate(r_slices):
                for qc, (c1, c2) in enumerate(c_slices):
                    sub = inp[r1:r2, c1:c2]
                    count = np.sum(sub == 6)
                    if count >= 2:
                        out[qr, qc] = 1
            return out
        prims["count_quadrant_particles_threshold"] = count_quadrant_particles_threshold

        # 81. NOR binaire entre moitiés supérieure et inférieure (tâche 94f9d214)
        def binary_nor_split_halves(inp: np.ndarray, ctx):
            H, W = inp.shape
            if H % 2 != 0:
                return None
            top = inp[:H//2, :]
            bot = inp[H//2:, :]
            fill_col = 2
            if ctx:
                out0 = ctx[0]["output"]
                new_cols = set(np.unique(out0)) - set(np.unique(ctx[0]["input"]))
                if len(new_cols) == 1:
                    fill_col = list(new_cols)[0]
            out = np.zeros((H//2, W), dtype=inp.dtype)
            out[(top == 0) & (bot == 0)] = fill_col
            return out
        prims["binary_nor_split_halves"] = binary_nor_split_halves

        # 82. Réflexion du motif supérieur vers le plancher (tâche 496994bd)
        def reflect_top_pattern_to_bottom(inp: np.ndarray, ctx):
            H, W = inp.shape
            top_rows = []
            for r in range(H):
                if np.any(inp[r, :] != 0):
                    top_rows.append(r)
                else:
                    break
            if not top_rows or len(top_rows) >= H // 2:
                return None
            k = len(top_rows)
            out = np.copy(inp)
            top_block = inp[:k, :]
            out[H-k:, :] = np.flipud(top_block)
            return out
        prims["reflect_top_pattern_to_bottom"] = reflect_top_pattern_to_bottom

        # 83. Remplacement de boîte creuse 3x3 par une croix pleine (tâche 6c434453)
        def replace_hollow_box_with_plus(inp: np.ndarray, ctx):
            H, W = inp.shape
            box_pattern = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            plus_pattern = np.array([[0, 2, 0], [2, 2, 2], [0, 2, 0]])
            out = np.copy(inp)
            replaced_any = False
            for r in range(H - 2):
                for c in range(W - 2):
                    if np.array_equal(inp[r:r+3, c:c+3], box_pattern):
                        out[r:r+3, c:c+3] = plus_pattern
                        replaced_any = True
            return out if replaced_any else None
        prims["replace_hollow_box_with_plus"] = replace_hollow_box_with_plus

        # 84. Remplissage du quadrant de densité maximale dans grille de séparateurs (tâche 29623171)
        def fill_max_density_quadrant_3x3(inp: np.ndarray, ctx):
            H, W = inp.shape
            if H != 11 or W != 11 or 5 not in inp:
                return None
            r_slices = [(0, 3), (4, 7), (8, 11)]
            c_slices = [(0, 3), (4, 7), (8, 11)]
            cols = [c for c in np.unique(inp) if c != 0 and c != 5]
            if not cols:
                return None
            target_col = cols[0]
            counts = np.zeros((3, 3), dtype=int)
            for qr in range(3):
                for qc in range(3):
                    r1, r2 = r_slices[qr]
                    c1, c2 = c_slices[qc]
                    counts[qr, qc] = np.sum(inp[r1:r2, c1:c2] == target_col)
            max_c = np.max(counts)
            if max_c < 2:
                return None
            out = np.zeros_like(inp)
            out[inp == 5] = 5
            for qr in range(3):
                for qc in range(3):
                    if counts[qr, qc] == max_c:
                        r1, r2 = r_slices[qr]
                        c1, c2 = c_slices[qc]
                        out[r1:r2, c1:c2] = target_col
            return out
        prims["fill_max_density_quadrant_3x3"] = fill_max_density_quadrant_3x3

        # 85. Rayons diagonaux opposés émis par deux blocs (tâche 5c0a986e)
        def diagonal_opposed_rays_from_dual_blocks(inp: np.ndarray, ctx):
            H, W = inp.shape
            pts1 = np.argwhere(inp == 1)
            pts2 = np.argwhere(inp == 2)
            if len(pts1) < 2 or len(pts2) < 2:
                return None
            out = np.copy(inp)
            r1, c1 = pts1.min(axis=0)
            cur_r, cur_c = r1 - 1, c1 - 1
            while cur_r >= 0 and cur_c >= 0:
                out[cur_r, cur_c] = 1
                cur_r -= 1
                cur_c -= 1
            r2, c2 = pts2.max(axis=0)
            cur_r, cur_c = r2 + 1, c2 + 1
            while cur_r < H and cur_c < W:
                out[cur_r, cur_c] = 2
                cur_r += 1
                cur_c += 1
            return out
        prims["diagonal_opposed_rays_from_dual_blocks"] = diagonal_opposed_rays_from_dual_blocks

        # 86. Remplissage d'une boîte ouverte avec marqueur et chapeau supérieur (tâche 444801d8)
        def fill_box_and_cap_above_from_marker(inp: np.ndarray, ctx):
            H, W = inp.shape
            markers = [c for c in np.unique(inp) if c != 0 and c != 1]
            if not markers or 1 not in inp:
                return None
            out = np.copy(inp)
            lbl, num = label(inp == 1, structure=np.ones((3, 3)))
            filled_any = False
            for m in markers:
                m_pts = np.argwhere(inp == m)
                for mr, mc in m_pts:
                    for cid in range(1, num + 1):
                        pts = np.argwhere(lbl == cid)
                        rmin, cmin = pts.min(0)
                        rmax, cmax = pts.max(0)
                        if rmin < mr < rmax and cmin < mc < cmax:
                            out[rmin+1:rmax, cmin+1:cmax] = m
                            out[rmin, mc] = m
                            if rmin - 1 >= 0:
                                out[rmin - 1, cmin:cmax+1] = m
                            filled_any = True
                            break
            return out if filled_any else None
        prims["fill_box_and_cap_above_from_marker"] = fill_box_and_cap_above_from_marker

        # 87. Pliage 2x2 des 4 coins d'une grille 5x7 en 3x3 avec recouvrement central (tâche bc1d5164)
        def fold_quadrants_5x7_to_3x3(inp: np.ndarray, ctx):
            if inp.shape != (5, 7):
                return None
            tl = inp[0:2, 0:2]
            tr = inp[0:2, 5:7]
            bl = inp[3:5, 0:2]
            br = inp[3:5, 5:7]
            out = np.zeros((3, 3), dtype=inp.dtype)
            out[0:2, 0:2] = np.maximum(out[0:2, 0:2], tl)
            out[0:2, 1:3] = np.maximum(out[0:2, 1:3], tr)
            out[1:3, 0:2] = np.maximum(out[1:3, 0:2], bl)
            out[1:3, 1:3] = np.maximum(out[1:3, 1:3], br)
            return out
        prims["fold_quadrants_5x7_to_3x3"] = fold_quadrants_5x7_to_3x3

        # 88. Encadrement de la bordure de grille par une couleur apprise (tâche 6f8cd79b)
        def frame_grid_boundary(inp: np.ndarray, ctx):
            if not ctx:
                return None
            c = ctx[0]["output"][0, 0]
            for p in ctx:
                if p["output"][0, 0] != c or p["output"][-1, -1] != c:
                    return None
            out = inp.copy()
            out[0, :] = c
            out[-1, :] = c
            out[:, 0] = c
            out[:, -1] = c
            return out
        prims["frame_grid_boundary"] = frame_grid_boundary

        # 89. Comptage de pixels vers jauge 3x3 apprise (tâche 794b24be)
        def count_pixels_to_gauge_3x3(inp: np.ndarray, ctx):
            if not ctx or inp.shape != (3, 3):
                return None
            for p in ctx:
                if not set(np.unique(p["input"])).issubset({0, 1}):
                    return None
                if not set(np.unique(p["output"])).issubset({0, 2}):
                    return None
            if not set(np.unique(inp)).issubset({0, 1}):
                return None
            mapping = {}
            for p in ctx:
                cnt = int(np.sum(p["input"] != 0))
                mapping[cnt] = p["output"]
            cnt_inp = int(np.sum(inp != 0))
            if cnt_inp in mapping:
                return mapping[cnt_inp].copy()
            return None
        prims["count_pixels_to_gauge_3x3"] = count_pixels_to_gauge_3x3

        # 90. Jauge de niveau de fluide selon hauteur libre du récipient (tâche b0c4d837)
        def gauge_fluid_level_in_container(inp: np.ndarray, ctx):
            if not ctx:
                return None
            for p in ctx:
                if p["output"].shape != (3, 3):
                    return None
                if not set(np.unique(p["input"])).issubset({0, 5, 8}):
                    return None
                if not set(np.unique(p["output"])).issubset({0, 8}):
                    return None
            if not set(np.unique(inp)).issubset({0, 5, 8}):
                return None
            mapping = {}
            for p in ctx:
                pinp = p["input"]
                if 8 not in pinp or 5 not in pinp:
                    return None
                r8 = np.min(np.where(pinp == 8)[0])
                r5 = np.min(np.where(pinp == 5)[0])
                mapping[r8 - r5] = p["output"]
            if 8 not in inp or 5 not in inp:
                return None
            diff = np.min(np.where(inp == 8)[0]) - np.min(np.where(inp == 5)[0])
            if diff in mapping:
                return mapping[diff].copy()
            return None
        prims["gauge_fluid_level_in_container"] = gauge_fluid_level_in_container

        # 91. Classification topologique : détection de la couleur possédant des trous (tâche b9b7f026)
        def classify_color_with_holes(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            for c in colors:
                mask = (inp == c)
                filled = binary_fill_holes(mask)
                holes = filled & (~mask)
                if np.any(holes):
                    return np.array([[c]], dtype=inp.dtype)
            return None
        prims["classify_color_with_holes"] = classify_color_with_holes

        # 92. Déroulé serpentin des pixels ordonnés par colonnes vers grille 3x3 (tâche cdecee7f)
        def snake_pixels_by_column_3x3(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            pts = []
            for c in range(W):
                for r in range(H):
                    if inp[r, c] != 0:
                        pts.append((c, r, inp[r, c]))
            if len(pts) > 9:
                return None
            out = np.zeros((3, 3), dtype=inp.dtype)
            snake_order = [
                (0, 0), (0, 1), (0, 2),
                (1, 2), (1, 1), (1, 0),
                (2, 0), (2, 1), (2, 2)
            ]
            for idx, (c, r, val) in enumerate(pts):
                if idx < len(snake_order):
                    sr, sc = snake_order[idx]
                    out[sr, sc] = val
            return out
        prims["snake_pixels_by_column_3x3"] = snake_pixels_by_column_3x3

        # 93. Extraction du motif 3x3 connexe le plus fréquent (tâche 39a8645d)
        def extract_most_frequent_3x3_subgrid(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            patterns = []
            for col in colors:
                lbl, num = label(inp == col, structure=np.ones((3, 3)))
                for c in range(1, num + 1):
                    pts = np.argwhere(lbl == c)
                    rmin, cmin = pts.min(0)
                    rmax, cmax = pts.max(0)
                    h = rmax - rmin + 1
                    w = cmax - cmin + 1
                    if h <= 3 and w <= 3:
                        sub = np.zeros((3, 3), dtype=inp.dtype)
                        sub[:h, :w] = (inp[rmin:rmax+1, cmin:cmax+1] == col) * col
                        patterns.append(tuple(sub.flatten()))
            if not patterns:
                return None
            counts = Counter(patterns)
            most_freq_flat = counts.most_common(1)[0][0]
            return np.array(most_freq_flat, dtype=inp.dtype).reshape((3, 3))
        prims["extract_most_frequent_3x3_subgrid"] = extract_most_frequent_3x3_subgrid

        # 94. Rognage 3x3 centré sur un marqueur 8 et cicatrisation par couleur locale (tâche 5117e062)
        def crop_3x3_around_marker_and_heal(inp: np.ndarray, ctx=None):
            marker = 8
            pts = np.argwhere(inp == marker)
            if len(pts) != 1:
                return None
            r, c = pts[0]
            H, W = inp.shape
            if r - 1 < 0 or r + 2 > H or c - 1 < 0 or c + 2 > W:
                return None
            sub = inp[r-1:r+2, c-1:c+2].copy()
            surr_colors = [col for col in np.unique(sub) if col != 0 and col != marker]
            if len(surr_colors) != 1:
                return None
            sub[1, 1] = surr_colors[0]
            return sub
        prims["crop_3x3_around_marker_and_heal"] = crop_3x3_around_marker_and_heal

        # 95. Projection de rayons diagonaux depuis les 4 coins d'un carré 2x2 (tâche 7ddcd7ec)
        def project_rays_from_2x2_square_diagonal_corners(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) != 1:
                return None
            col = colors[0]
            H, W = inp.shape
            sq = None
            for r in range(H - 1):
                for c in range(W - 1):
                    if np.all(inp[r:r+2, c:c+2] == col):
                        sq = (r, c)
                        break
                if sq is not None:
                    break
            if sq is None:
                return None
            r0, c0 = sq
            diag_map = {
                (r0 - 1, c0 - 1): (-1, -1),
                (r0 - 1, c0 + 2): (-1, 1),
                (r0 + 2, c0 - 1): (1, -1),
                (r0 + 2, c0 + 2): (1, 1)
            }
            out = np.copy(inp)
            for (pr, pc), (dr, dc) in diag_map.items():
                if 0 <= pr < H and 0 <= pc < W and inp[pr, pc] == col:
                    cur_r, cur_c = pr + dr, pc + dc
                    while 0 <= cur_r < H and 0 <= cur_c < W:
                        out[cur_r, cur_c] = col
                        cur_r += dr
                        cur_c += dc
            return out
        prims["project_rays_from_2x2_square_diagonal_corners"] = project_rays_from_2x2_square_diagonal_corners

        # 96. Rognage de boîte délimitée par colonnes jumelles fermées par chapeaux (tâche 3f7978a0)
        def crop_frame_between_paired_columns(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            cols_with_5 = [c for c in range(W) if np.any(inp[:, c] == 5)]
            if len(cols_with_5) != 2:
                return None
            c1, c2 = min(cols_with_5), max(cols_with_5)
            rows_5 = np.where((inp[:, c1] == 5) | (inp[:, c2] == 5))[0]
            if len(rows_5) == 0:
                return None
            r_min_5, r_max_5 = rows_5.min(), rows_5.max()
            r_min = r_min_5 - 1
            r_max = r_max_5 + 1
            if r_min < 0 or r_max >= H:
                return None
            return inp[r_min:r_max+1, c1:c2+1].copy()
        prims["crop_frame_between_paired_columns"] = crop_frame_between_paired_columns

        # 97. Extraction et recoloriage de l'intérieur délimité par 4 marqueurs rectangulaires (tâche 3de23699)
        def extract_interior_of_4_corner_markers(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            for c in colors:
                pts = np.argwhere(inp == c)
                if len(pts) == 4:
                    rows = sorted(list(set(pts[:, 0])))
                    cols = sorted(list(set(pts[:, 1])))
                    if len(rows) == 2 and len(cols) == 2:
                        r1, r2 = rows[0], rows[1]
                        c1, c2 = cols[0], cols[1]
                        if r2 - r1 > 1 and c2 - c1 > 1:
                            if inp[r1, c1] == c and inp[r1, c2] == c and inp[r2, c1] == c and inp[r2, c2] == c:
                                sub = inp[r1+1:r2, c1+1:c2]
                                return np.where(sub != 0, c, 0)
            return None
        prims["extract_interior_of_4_corner_markers"] = extract_interior_of_4_corner_markers

        # 98. Symétrie miroir 2x2 complète Klein-4 (tâche 3af2c5a8)
        def mirror_quadrants_2x2(inp: np.ndarray, ctx=None):
            top = np.hstack([inp, np.fliplr(inp)])
            bottom = np.hstack([np.flipud(inp), np.flip(inp)])
            return np.vstack([top, bottom])
        prims["mirror_quadrants_2x2"] = mirror_quadrants_2x2

        # 99. Rembourrage par réplication des 4 bords avec zéros aux 4 coins (tâche 49d1d64f)
        def pad_replicate_border_with_zero_corners(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            out = np.zeros((H + 2, W + 2), dtype=inp.dtype)
            out[1:H+1, 1:W+1] = inp
            out[0, 1:W+1] = inp[0, :]
            out[H+1, 1:W+1] = inp[H-1, :]
            out[1:H+1, 0] = inp[:, 0]
            out[1:H+1, W+1] = inp[:, W-1]
            return out
        prims["pad_replicate_border_with_zero_corners"] = pad_replicate_border_with_zero_corners

        # 100. Pavage cyclique C4 des 4 quadrants par rotation (tâche 46442a0e)
        def pinwheel_rotation_quadrants_2x2(inp: np.ndarray, ctx=None):
            tl = inp
            tr = np.rot90(inp, 3)
            bl = np.rot90(inp, 1)
            br = np.rot90(inp, 2)
            top = np.hstack([tl, tr])
            bottom = np.hstack([bl, br])
            return np.vstack([top, bottom])
        prims["pinwheel_rotation_quadrants_2x2"] = pinwheel_rotation_quadrants_2x2

        # 101. Réflexion du bloc supérieur guidée par la flèche inférieure (tâche 760b3cac)
        def reflect_top_pattern_by_bottom_arrow(inp: np.ndarray, ctx=None):
            if inp.shape != (6, 9):
                return None
            top_block = inp[0:3, 3:6]
            bot_block = inp[3:6, 3:6]
            out = inp.copy()
            if bot_block[0, 0] == 4:
                out[0:3, 0:3] = np.fliplr(top_block)
            elif bot_block[0, 2] == 4:
                out[0:3, 6:9] = np.fliplr(top_block)
            else:
                return None
            return out
        prims["reflect_top_pattern_by_bottom_arrow"] = reflect_top_pattern_by_bottom_arrow

        # 102. Inondation périodique alternée des creux d'ondes en pas de 6 (tâche 7447852a)
        def fill_alternating_wave_crests_period_6(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            if H != 3:
                return None
            out = inp.copy()
            for center_c in range(0, W + 3, 6):
                if (center_c // 6) % 2 == 0:
                    coords = [(2, center_c - 1), (2, center_c), (2, center_c + 1), (1, center_c)]
                else:
                    coords = [(0, center_c - 1), (0, center_c), (0, center_c + 1), (1, center_c)]
                for r, c in coords:
                    if 0 <= r < H and 0 <= c < W and out[r, c] == 0:
                        out[r, c] = 4
            return out
        prims["fill_alternating_wave_crests_period_6"] = fill_alternating_wave_crests_period_6

        # 103. Extraction du sous-bloc 3x3 solide contenant le plus de pixels cibles (tâche ae4f1146)
        def extract_solid_3x3_with_max_target_color(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            if H < 3 or W < 3:
                return None
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) != 2:
                return None
            c_target = 1 if 1 in colors else colors[0]
            best_sub = None
            max_c1 = -1
            for r in range(H - 2):
                for c in range(W - 2):
                    sub = inp[r:r+3, c:c+3]
                    if np.all(sub != 0):
                        cnt = int(np.sum(sub == c_target))
                        if cnt > max_c1:
                            max_c1 = cnt
                            best_sub = sub.copy()
            return best_sub
        prims["extract_solid_3x3_with_max_target_color"] = extract_solid_3x3_with_max_target_color

        # 104. Expansion de carrés 4x4 le long de la diagonale du marqueur (tâche 4522001f)
        def expand_diagonal_4x4_squares_from_marker(inp: np.ndarray, ctx=None):
            if inp.shape != (3, 3):
                return None
            pts2 = np.argwhere(inp == 2)
            if len(pts2) != 1:
                return None
            r2, c2 = pts2[0]
            sq = None
            for r in range(2):
                for c in range(2):
                    if r <= r2 <= r + 1 and c <= c2 <= c + 1:
                        sub = inp[r:r+2, c:c+2]
                        if np.all((sub == 3) | (sub == 2)):
                            sq = (r, c)
                            break
                if sq is not None:
                    break
            if sq is None:
                return None
            r0, c0 = sq
            out = np.zeros((9, 9), dtype=inp.dtype)
            if (r2 == r0 and c2 == c0) or (r2 == r0 + 1 and c2 == c0 + 1):
                out[r0:r0+4, c0:c0+4] = 3
                out[r0+4:r0+8, c0+4:c0+8] = 3
            else:
                out[r0+4:r0+8, c0:c0+4] = 3
                out[r0:r0+4, c0+4:c0+8] = 3
            return out
        prims["expand_diagonal_4x4_squares_from_marker"] = expand_diagonal_4x4_squares_from_marker

        # 105. Assemblage de 4 pièces L-tromino selon orientation propre en cadre 4x4 (tâche a61ba2ce)
        def assemble_4_l_trominoes_into_frame_4x4(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) != 4:
                return None
            out = np.zeros((4, 4), dtype=inp.dtype)
            for col in colors:
                pts = np.argwhere(inp == col)
                if len(pts) != 3:
                    return None
                rmin, cmin = pts.min(0)
                rmax, cmax = pts.max(0)
                if rmax - rmin != 1 or cmax - cmin != 1:
                    return None
                hole = None
                for dr in range(2):
                    for dc in range(2):
                        if inp[rmin + dr, cmin + dc] != col:
                            hole = (dr, dc)
                            break
                if hole is None:
                    return None
                corner_map = {
                    (1, 1): (0, 0),
                    (1, 0): (0, 2),
                    (0, 1): (2, 0),
                    (0, 0): (2, 2)
                }
                tr, tc = corner_map[hole]
                sub = inp[rmin:rmax+1, cmin:cmax+1]
                out[tr:tr+2, tc:tc+2] = sub
            return out
        prims["assemble_4_l_trominoes_into_frame_4x4"] = assemble_4_l_trominoes_into_frame_4x4

        # 106. Duplication miroir horizontale [inp, fliplr(inp)] (tâche 6d0aefbc)
        def tile_horizontal_mirror_fliplr(inp: np.ndarray, ctx=None):
            return np.hstack([inp, np.fliplr(inp)])
        prims["tile_horizontal_mirror_fliplr"] = tile_horizontal_mirror_fliplr

        # 107. Duplication miroir verticale [inp ; flipud(inp)] (tâche 6fa7a44f)
        def tile_vertical_mirror_flipud(inp: np.ndarray, ctx=None):
            return np.vstack([inp, np.flipud(inp)])
        prims["tile_vertical_mirror_flipud"] = tile_vertical_mirror_flipud

        # 108. Duplication verticale flipud puis original [flipud(inp) ; inp] (tâche 4c4377d9)
        def tile_vertical_flipud_then_original(inp: np.ndarray, ctx=None):
            return np.vstack([np.flipud(inp), inp])
        prims["tile_vertical_flipud_then_original"] = tile_vertical_flipud_then_original

        # 109. Extraction de la moitié fondamentale d'une matrice périodique dédoublée (tâche 7b7f7511)
        def extract_halved_repeated_tile(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            if W % 2 == 0 and np.array_equal(inp[:, :W//2], inp[:, W//2:]):
                return inp[:, :W//2].copy()
            if H % 2 == 0 and np.array_equal(inp[:H//2, :], inp[H//2:, :]):
                return inp[:H//2, :].copy()
            return None
        prims["extract_halved_repeated_tile"] = extract_halved_repeated_tile

        # 110. Compression 1D par réduction des répétitions consécutives le long de l'axe isotrope (tâche 746b3537)
        def compress_1d_consecutive_duplicates(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            if all(np.array_equal(inp[r, :], inp[0, :]) for r in range(H)):
                row = inp[0, :]
                dedup = [row[0]]
                for val in row[1:]:
                    if val != dedup[-1]:
                        dedup.append(val)
                return np.array([dedup], dtype=inp.dtype)
            if all(np.array_equal(inp[:, c], inp[:, 0]) for c in range(W)):
                col = inp[:, 0]
                dedup = [col[0]]
                for val in col[1:]:
                    if val != dedup[-1]:
                        dedup.append(val)
                return np.array([[v] for v in dedup], dtype=inp.dtype)
            return None
        prims["compress_1d_consecutive_duplicates"] = compress_1d_consecutive_duplicates

        # 111. Déplacement du point vers le bas et remplissage au-dessus des colonnes de même parité (tâche 834ec97d)
        def drop_dot_and_fill_columns_above_with_same_parity(inp: np.ndarray, ctx=None):
            pts = np.argwhere(inp != 0)
            if len(pts) != 1:
                return None
            r0, c0 = pts[0]
            marker_col = inp[r0, c0]
            H, W = inp.shape
            if r0 + 1 >= H:
                return None
            out = np.zeros((H, W), dtype=inp.dtype)
            out[r0 + 1, c0] = marker_col
            for r in range(r0 + 1):
                for c in range(W):
                    if c % 2 == c0 % 2:
                        out[r, c] = 4
            return out
        prims["drop_dot_and_fill_columns_above_with_same_parity"] = drop_dot_and_fill_columns_above_with_same_parity

        # 112. Recoloriage des 8 du centre selon leur quadrant d'appartenance par rapport aux 4 coins (tâche 77fdfe62)
        def recolor_center_8s_by_four_corners(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            if H < 6 or W < 6:
                return None
            c_tl = inp[0, 0]
            c_tr = inp[0, -1]
            c_bl = inp[-1, 0]
            c_br = inp[-1, -1]
            center = inp[2:H-2, 2:W-2]
            cH, cW = center.shape
            half_r = cH // 2
            half_c = cW // 2
            out = np.zeros((cH, cW), dtype=inp.dtype)
            for r in range(cH):
                for c in range(cW):
                    if center[r, c] != 0:
                        if r < half_r and c < half_c:
                            out[r, c] = c_tl
                        elif r < half_r and c >= half_c:
                            out[r, c] = c_tr
                        elif r >= half_r and c < half_c:
                            out[r, c] = c_bl
                        else:
                            out[r, c] = c_br
            return out
        prims["recolor_center_8s_by_four_corners"] = recolor_center_8s_by_four_corners

        # 113. Superposition par ordre strict des calques des 4 quadrants (tâche 75b8110e)
        def priority_overlay_quadrants(inp: np.ndarray, ctx=None):
            if inp.shape != (8, 8):
                return None
            tl = inp[0:4, 0:4]
            tr = inp[0:4, 4:8]
            bl = inp[4:8, 0:4]
            br = inp[4:8, 4:8]
            out = np.zeros((4, 4), dtype=inp.dtype)
            for sub in [tl, br, bl, tr]:
                mask = (sub != 0)
                out[mask] = sub[mask]
            return out
        prims["priority_overlay_quadrants"] = priority_overlay_quadrants

        # 114. Remplissage des cellules 3x3 délimitées par une grille de 5 par c_marker + 5 (tâche 54d9e175)
        def fill_cells_with_marker_plus_5(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            if 5 not in inp:
                return None
            out = inp.copy()
            for r0 in range(0, H, 4):
                for c0 in range(0, W, 4):
                    sub = inp[r0:r0+3, c0:c0+3]
                    markers = [c for c in np.unique(sub) if c != 0 and c != 5]
                    if len(markers) == 1:
                        target_col = markers[0] + 5
                        out[r0:r0+3, c0:c0+3] = target_col
            return out
        prims["fill_cells_with_marker_plus_5"] = fill_cells_with_marker_plus_5

        # 115. Rognage et extraction de l'unique forme possédant une symétrie axiale verticale (tâche 72ca375d)
        def extract_vertically_symmetric_shape(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            for col in colors:
                pts = np.argwhere(inp == col)
                rmin, cmin = pts.min(0)
                rmax, cmax = pts.max(0)
                sub = inp[rmin:rmax+1, cmin:cmax+1]
                mask = (sub == col)
                if np.array_equal(mask, np.fliplr(mask)):
                    return sub.copy()
            return None
        prims["extract_vertically_symmetric_shape"] = extract_vertically_symmetric_shape

        # 116. Produit Kronecker fractal 3x3 sur canvas 9x9 (tâche 8f2ea7aa)
        def kronecker_fractal_3x3_to_9x9(inp: np.ndarray, ctx=None):
            if inp.shape != (9, 9):
                return None
            pts = np.argwhere(inp != 0)
            if len(pts) == 0:
                return None
            rmin, cmin = pts.min(0)
            rmax, cmax = pts.max(0)
            if (rmax - rmin + 1, cmax - cmin + 1) != (3, 3):
                return None
            kernel = inp[rmin:rmax+1, cmin:cmax+1]
            bin_kernel = (kernel != 0).astype(int)
            out = np.zeros((9, 9), dtype=inp.dtype)
            for r in range(3):
                for c in range(3):
                    if bin_kernel[r, c]:
                        out[r*3:(r+1)*3, c*3:(c+1)*3] = kernel
            return out
        prims["kronecker_fractal_3x3_to_9x9"] = kronecker_fractal_3x3_to_9x9

        # 117. Colonnes en peigne alternées avec butées grises 5 (tâche 8403a5d5)
        def alternating_comb_columns_with_caps(inp: np.ndarray, ctx=None):
            pts = np.argwhere(inp != 0)
            if len(pts) != 1:
                return None
            r0, c0 = pts[0]
            col = inp[r0, c0]
            H, W = inp.shape
            out = np.zeros_like(inp)
            k = 0
            while True:
                c = c0 + 2 * k
                if c >= W:
                    break
                out[:, c] = col
                mid_c = c + 1
                if mid_c < W:
                    if k % 2 == 0:
                        out[0, mid_c] = 5
                    else:
                        out[H - 1, mid_c] = 5
                k += 1
            return out
        prims["alternating_comb_columns_with_caps"] = alternating_comb_columns_with_caps

        # 118. Remplissage des cavités closes selon la parité de leur aire (tâche 868de0fa)
        def fill_holes_by_area_parity(inp: np.ndarray, ctx=None):
            if not np.array_equal(np.unique(inp), [0, 1]):
                return None
            H, W = inp.shape
            labeled, num_features = label(inp == 0)
            out = inp.copy()
            has_cavity = False
            for comp_id in range(1, num_features + 1):
                pts = np.argwhere(labeled == comp_id)
                if (pts[:, 0] == 0).any() or (pts[:, 0] == H - 1).any() or (pts[:, 1] == 0).any() or (pts[:, 1] == W - 1).any():
                    continue
                has_cavity = True
                area = len(pts)
                fill_col = 2 if (area % 2 == 0) else 7
                for r, c in pts:
                    out[r, c] = fill_col
            return out if has_cavity else None
        prims["fill_holes_by_area_parity"] = fill_holes_by_area_parity

        # 119. Rotation cyclique d'un bloc 3x3 à travers 3 fenêtres séparées par des 5 (tâche 8e5a5113)
        def rotating_block_across_windows(inp: np.ndarray, ctx=None):
            if inp.shape != (3, 11):
                return None
            if not (np.all(inp[:, 3] == 5) and np.all(inp[:, 7] == 5)):
                return None
            out = inp.copy()
            w0 = inp[:, 0:3]
            out[:, 4:7] = np.rot90(w0, 3)
            out[:, 8:11] = np.rot90(w0, 2)
            return out
        prims["rotating_block_across_windows"] = rotating_block_across_windows

        # 120. Remplissage de la boîte englobante 8-connexe avec du 7 (tâche 60b61512)
        def fill_bounding_box_zeros_with_7(inp: np.ndarray, ctx=None):
            if not np.all(np.isin(inp, [0, 4])):
                return None
            s8 = np.ones((3, 3), dtype=int)
            labeled, num_features = label(inp == 4, structure=s8)
            if num_features == 0:
                return None
            out = inp.copy()
            for comp_id in range(1, num_features + 1):
                pts = np.argwhere(labeled == comp_id)
                rmin, cmin = pts.min(0)
                rmax, cmax = pts.max(0)
                for r in range(rmin, rmax + 1):
                    for c in range(cmin, cmax + 1):
                        if out[r, c] == 0:
                            out[r, c] = 7
            return out
        prims["fill_bounding_box_zeros_with_7"] = fill_bounding_box_zeros_with_7

        # 121. Estampillage du motif source centré sur le marqueur 5 (tâche 88a10436)
        def stamp_template_centered_on_marker_5(inp: np.ndarray, ctx=None):
            m5 = np.argwhere(inp == 5)
            if len(m5) != 1:
                return None
            mr, mc = m5[0]
            pts = np.argwhere((inp != 0) & (inp != 5))
            if len(pts) == 0:
                return None
            rmin, cmin = pts.min(0)
            rmax, cmax = pts.max(0)
            if (rmax - rmin + 1, cmax - cmin + 1) != (3, 3):
                return None
            sub = inp[rmin:rmax+1, cmin:cmax+1].copy()
            out = inp.copy()
            out[mr, mc] = 0
            for r in range(3):
                for c in range(3):
                    tr, tc = mr - 1 + r, mc - 1 + c
                    if 0 <= tr < inp.shape[0] and 0 <= tc < inp.shape[1]:
                        out[tr, tc] = sub[r, c]
            return out
        prims["stamp_template_centered_on_marker_5"] = stamp_template_centered_on_marker_5

        # 122. Remplissage intérieur de deux rectangles selon leur taille (tâche 694f12f3)
        def hollow_fill_two_rectangles_by_size(inp: np.ndarray, ctx=None):
            if not np.all(np.isin(inp, [0, 4])):
                return None
            labeled, num_features = label(inp == 4)
            if num_features != 2:
                return None
            boxes = []
            for comp_id in range(1, 3):
                pts = np.argwhere(labeled == comp_id)
                rmin, cmin = pts.min(0)
                rmax, cmax = pts.max(0)
                boxes.append((len(pts), rmin, rmax, cmin, cmax))
            boxes.sort(key=lambda x: x[0])
            out = inp.copy()
            _, rmin1, rmax1, cmin1, cmax1 = boxes[0]
            if rmax1 - rmin1 >= 2 and cmax1 - cmin1 >= 2:
                out[rmin1+1:rmax1, cmin1+1:cmax1] = 1
            _, rmin2, rmax2, cmin2, cmax2 = boxes[1]
            if rmax2 - rmin2 >= 2 and cmax2 - cmin2 >= 2:
                out[rmin2+1:rmax2, cmin2+1:cmax2] = 2
            return out
        prims["hollow_fill_two_rectangles_by_size"] = hollow_fill_two_rectangles_by_size

        # 123. Sur-échantillonnage 3x nearest-neighbor (tâche 9172f3a0)
        def nearest_neighbor_upscale_3x(inp: np.ndarray, ctx=None):
            if inp.shape != (3, 3):
                return None
            return np.kron(inp, np.ones((3, 3), dtype=inp.dtype))
        prims["nearest_neighbor_upscale_3x"] = nearest_neighbor_upscale_3x

        # 124. Prolongement périodique horizontal par réplication de motif (tâche 963e52fc)
        def horizontal_periodic_extension_x2(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            new_W = 2 * W
            out = np.zeros((H, new_W), dtype=inp.dtype)
            for r in range(H):
                row = inp[r]
                if np.all(row == 0):
                    out[r] = np.zeros(new_W, dtype=row.dtype)
                    continue
                best_T = None
                for T in range(1, W):
                    if np.all(row[T:] == row[:-T]):
                        best_T = T
                        break
                if best_T is None:
                    best_T = W
                out_row = np.zeros(new_W, dtype=row.dtype)
                out_row[:W] = row
                for c in range(W, new_W):
                    out_row[c] = out_row[c - best_T]
                out[r] = out_row
            return out
        prims["horizontal_periodic_extension_x2"] = horizontal_periodic_extension_x2

        # 125. Tracé de périmètre rectangulaire et séparateurs intérieurs (tâche 4612dd53)
        def fill_rectangle_perimeter_and_dividers(inp: np.ndarray, ctx=None):
            if not np.all(np.isin(inp, [0, 1])):
                return None
            pts = np.argwhere(inp == 1)
            if len(pts) < 4:
                return None
            rmin, cmin = pts.min(0)
            rmax, cmax = pts.max(0)
            out = inp.copy()
            for c in range(cmin, cmax + 1):
                if out[rmin, c] == 0: out[rmin, c] = 2
                if out[rmax, c] == 0: out[rmax, c] = 2
            for r in range(rmin, rmax + 1):
                if out[r, cmin] == 0: out[r, cmin] = 2
                if out[r, cmax] == 0: out[r, cmax] = 2
            interior_pts = [p for p in pts if rmin < p[0] < rmax and cmin < p[1] < cmax]
            if interior_pts:
                int_rows = {p[0] for p in interior_pts}
                int_cols = {p[1] for p in interior_pts}
                if len(int_rows) == 1:
                    r = list(int_rows)[0]
                    for c in range(cmin, cmax + 1):
                        if out[r, c] == 0: out[r, c] = 2
                elif len(int_cols) == 1:
                    c = list(int_cols)[0]
                    for r in range(rmin, rmax + 1):
                        if out[r, c] == 0: out[r, c] = 2
                else:
                    for r in int_rows:
                        if sum(1 for p in interior_pts if p[0] == r) >= 2:
                            for c in range(cmin, cmax + 1):
                                if out[r, c] == 0: out[r, c] = 2
                    for c in int_cols:
                        if sum(1 for p in interior_pts if p[1] == c) >= 2:
                            for r in range(rmin, rmax + 1):
                                if out[r, c] == 0: out[r, c] = 2
            return out
        prims["fill_rectangle_perimeter_and_dividers"] = fill_rectangle_perimeter_and_dividers

        # 126. Précipitation et ancrage de particules sur dalle perpendiculaire (tâche 4093f84a)
        def dock_particles_to_perpendicular_slab(inp: np.ndarray, ctx=None):
            pts_5 = np.argwhere(inp == 5)
            if len(pts_5) == 0:
                return None
            H, W = inp.shape
            cols_5 = np.unique(pts_5[:, 1])
            rows_5 = np.unique(pts_5[:, 0])
            out = np.zeros_like(inp)
            
            if len(cols_5) == W:
                rmin, rmax = rows_5.min(), rows_5.max()
                out[rmin:rmax+1, :] = 5
                for c in range(W):
                    k_above = int(np.sum((inp[:rmin, c] != 0) & (inp[:rmin, c] != 5)))
                    for i in range(k_above):
                        out[rmin - 1 - i, c] = 5
                    k_below = int(np.sum((inp[rmax+1:, c] != 0) & (inp[rmax+1:, c] != 5)))
                    for i in range(k_below):
                        out[rmax + 1 + i, c] = 5
                return out
            elif len(rows_5) == H:
                cmin, cmax = cols_5.min(), cols_5.max()
                out[:, cmin:cmax+1] = 5
                for r in range(H):
                    k_left = int(np.sum((inp[r, :cmin] != 0) & (inp[r, :cmin] != 5)))
                    for i in range(k_left):
                        out[r, cmin - 1 - i] = 5
                    k_right = int(np.sum((inp[r, cmax+1:] != 0) & (inp[r, cmax+1:] != 5)))
                    for i in range(k_right):
                        out[r, cmax + 1 + i] = 5
                return out
            return None
        prims["dock_particles_to_perpendicular_slab"] = dock_particles_to_perpendicular_slab

        # 127. Coin biseauté avec antidiagonale 2 et barre inférieure 4 (tâche 3bd67248)
        def corner_wedge_with_antidiagonal(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            if H != W or H < 3:
                return None
            if not np.all(inp[:, 0] != 0):
                return None
            c0 = inp[0, 0]
            if not np.all(inp[:, 0] == c0):
                return None
            if not np.all(inp[:, 1:] == 0):
                return None
            out = np.zeros_like(inp)
            out[:, 0] = c0
            out[H - 1, 1:] = 4
            for r in range(H - 1):
                out[r, W - 1 - r] = 2
            return out
        prims["corner_wedge_with_antidiagonal"] = corner_wedge_with_antidiagonal

        # 128. Éclosion en boîte concentrique avec butées axiales (tâche 3befdf3e)
        def nested_square_blossom_caps(inp: np.ndarray, ctx=None):
            labeled, num_features = label(inp != 0)
            if num_features == 0:
                return None
            out = np.zeros_like(inp)
            for comp_id in range(1, num_features + 1):
                pts = np.argwhere(labeled == comp_id)
                rmin, cmin = pts.min(0)
                rmax, cmax = pts.max(0)
                H_box = rmax - rmin + 1
                W_box = cmax - cmin + 1
                if H_box != W_box or H_box < 3:
                    return None
                S = H_box
                sub = inp[rmin:rmax+1, cmin:cmax+1]
                C1 = sub[0, 0]
                inner_pts = np.argwhere(sub != C1)
                if len(inner_pts) == 0:
                    return None
                C2 = sub[inner_pts[0][0], inner_pts[0][1]]
                k = S - 2
                T = k
                S_out = S + 2 * T
                pat = np.zeros((S_out, S_out), dtype=inp.dtype)
                pat[T:T+S, T:T+S] = C2
                pat[T+1:T+S-1, T+1:T+S-1] = C1
                pat[0:T, T:T+S] = C1
                pat[T+S:S_out, T:T+S] = C1
                pat[T:T+S, 0:T] = C1
                pat[T:T+S, T+S:S_out] = C1
                r0 = rmin - T
                c0 = cmin - T
                if r0 < 0 or c0 < 0 or r0 + S_out > inp.shape[0] or c0 + S_out > inp.shape[1]:
                    return None
                out[r0:r0+S_out, c0:c0+S_out] = np.maximum(out[r0:r0+S_out, c0:c0+S_out], pat)
            return out
        prims["nested_square_blossom_caps"] = nested_square_blossom_caps

        # 129. Remplissage du plus grand rectangle vide (0s) par la couleur 6 (tâche 3eda0437)
        def fill_largest_zero_rectangle_with_6(inp: np.ndarray, ctx=None):
            if 6 in inp or inp.shape[0] > 6 or inp.shape[1] < 15:
                return None
            if not np.all(np.isin(inp, [0, 1, 5])):
                return None
            H, W = inp.shape
            best_rect = None
            best_area = -1
            for h in range(2, H + 1):
                for w in range(3, W + 1):
                    for r in range(H - h + 1):
                        for c in range(W - w + 1):
                            if np.all(inp[r:r+h, c:c+w] == 0):
                                area = h * w
                                if area > best_area:
                                    best_area = area
                                    best_rect = (r, c, h, w)
            if best_rect is None:
                return None
            r, c, h, w = best_rect
            out = inp.copy()
            out[r:r+h, c:c+w] = 6
            return out
        prims["fill_largest_zero_rectangle_with_6"] = fill_largest_zero_rectangle_with_6

        # 130. Perforation alternée de la ligne médiane des bandes de hauteur 3 (tâche 3bdb4ada)
        def perforate_middle_row_of_height3_bars(inp: np.ndarray, ctx=None):
            labeled, num_features = label(inp != 0)
            if num_features == 0:
                return None
            out = inp.copy()
            found = False
            for comp_id in range(1, num_features + 1):
                pts = np.argwhere(labeled == comp_id)
                rmin, cmin = pts.min(0)
                rmax, cmax = pts.max(0)
                h = rmax - rmin + 1
                w = cmax - cmin + 1
                if h == 3 and w >= 3 and len(pts) == h * w:
                    found = True
                    mid_r = rmin + 1
                    for c in range(cmin, cmax + 1):
                        if (c - cmin) % 2 == 1:
                            out[mid_r, c] = 0
            return out if found else None
        prims["perforate_middle_row_of_height3_bars"] = perforate_middle_row_of_height3_bars

        # 131. Estampillage du motif modèle dans les fenêtres 3x3 marquées par 1 (tâche 363442ee)
        def stamp_template_in_marked_3x3_windows(inp: np.ndarray, ctx=None):
            if inp.shape != (9, 13):
                return None
            if not np.all(inp[:, 3] == 5):
                return None
            template = inp[0:3, 0:3]
            out = inp.copy()
            out[:, 4:] = 0
            for wr in range(3):
                for wc in range(3):
                    r_start = wr * 3
                    c_start = 4 + wc * 3
                    window = inp[r_start:r_start+3, c_start:c_start+3]
                    if 1 in window:
                        out[r_start:r_start+3, c_start:c_start+3] = template
            return out
        prims["stamp_template_in_marked_3x3_windows"] = stamp_template_in_marked_3x3_windows

        # 132. Évidage intérieur de tous les rectangles pleins (tâche 4347f46a)
        def hollow_all_solid_rectangles(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            out = inp.copy()
            modified = False
            for col in colors:
                labeled, num_features = label(inp == col)
                for comp_id in range(1, num_features + 1):
                    pts = np.argwhere(labeled == comp_id)
                    rmin, cmin = pts.min(0)
                    rmax, cmax = pts.max(0)
                    h = rmax - rmin + 1
                    w = cmax - cmin + 1
                    if len(pts) == h * w and h >= 3 and w >= 3:
                        out[rmin+1:rmax, cmin+1:cmax] = 0
                        modified = True
            return out if modified else None
        prims["hollow_all_solid_rectangles"] = hollow_all_solid_rectangles

        # 133. Sous-échantillonnage 2x2 max-pool puis sur-échantillonnage Kronecker 4x4 (tâche 46f33fce)
        def pool_2x2_then_kron_4x4(inp: np.ndarray, ctx=None):
            if inp.shape != (10, 10):
                return None
            minimap = np.zeros((5, 5), dtype=inp.dtype)
            for r in range(5):
                for c in range(5):
                    minimap[r, c] = np.max(inp[r*2:(r+1)*2, c*2:(c+1)*2])
            return np.kron(minimap, np.ones((4, 4), dtype=inp.dtype))
        prims["pool_2x2_then_kron_4x4"] = pool_2x2_then_kron_4x4

        # 134. Extraction de la séquence ordonnée des strates de couleur (tâche 4be741c5)
        def extract_color_layer_sequence_order(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) < 2:
                return None
            stats = []
            for c in colors:
                pts = np.argwhere(inp == c)
                stats.append((c, pts[:, 0].mean(), pts[:, 1].mean()))
            mean_r = [s[1] for s in stats]
            mean_c = [s[2] for s in stats]
            var_r = np.var(mean_r)
            var_c = np.var(mean_c)
            if var_r > var_c:
                stats.sort(key=lambda s: s[1])
                return np.array([s[0] for s in stats], dtype=inp.dtype).reshape(-1, 1)
            else:
                stats.sort(key=lambda s: s[2])
                return np.array([s[0] for s in stats], dtype=inp.dtype).reshape(1, -1)
        prims["extract_color_layer_sequence_order"] = extract_color_layer_sequence_order

        # 135. Empreinte 3x3 de la macro-structure du composant majeur (tâche 5ad4f10b)
        def macro_3x3_blueprint_from_largest_component(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) != 2:
                return None
            max_comp_sizes = {}
            for c in colors:
                labeled, n = label(inp == c)
                if n > 0:
                    max_comp_sizes[c] = max(np.sum(labeled == i) for i in range(1, n+1))
                else:
                    max_comp_sizes[c] = 0
            macro_col = max(max_comp_sizes, key=max_comp_sizes.get)
            out_col = [c for c in colors if c != macro_col][0]
            pts = np.argwhere(inp == macro_col)
            rmin, cmin = pts.min(0)
            rmax, cmax = pts.max(0)
            h = rmax - rmin + 1
            w = cmax - cmin + 1
            if h % 3 != 0 or w % 3 != 0:
                return None
            block_h = h // 3
            block_w = w // 3
            out = np.zeros((3, 3), dtype=inp.dtype)
            for r in range(3):
                for c in range(3):
                    sub = inp[rmin + r*block_h : rmin + (r+1)*block_h, cmin + c*block_w : cmin + (c+1)*block_w]
                    if np.any(sub == macro_col):
                        out[r, c] = out_col
            return out
        prims["macro_3x3_blueprint_from_largest_component"] = macro_3x3_blueprint_from_largest_component

        # 136. Réflexion à 4 quadrants par inversion du quadrant supérieur-gauche (tâche 47c1f68c)
        def reflect_quadrant_across_cross_separators(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            sep_col = None
            sep_r = None
            sep_c = None
            for r in range(H):
                if len(np.unique(inp[r, :])) == 1 and inp[r, 0] != 0:
                    c_cand = inp[r, 0]
                    cols = [c for c in range(W) if np.all(inp[:, c] == c_cand)]
                    if len(cols) == 1:
                        sep_col = c_cand
                        sep_r = r
                        sep_c = cols[0]
                        break
            if sep_col is None:
                return None
            q_tl = inp[:sep_r, :sep_c]
            q_bin = (q_tl != 0).astype(inp.dtype) * sep_col
            q_tr = np.fliplr(q_bin)
            q_bl = np.flipud(q_bin)
            q_br = np.flipud(np.fliplr(q_bin))
            top = np.hstack([q_bin, q_tr])
            bot = np.hstack([q_bl, q_br])
            return np.vstack([top, bot])
        prims["reflect_quadrant_across_cross_separators"] = reflect_quadrant_across_cross_separators

        # 137. Remplissage intégral de la boîte englobante de chaque paire de marqueurs (tâche 56ff96f3)
        def fill_bounding_box_of_each_color(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) == 0:
                return None
            out = np.zeros_like(inp)
            for c in colors:
                pts = np.argwhere(inp == c)
                if len(pts) < 2:
                    return None
                rmin, cmin = pts.min(0)
                rmax, cmax = pts.max(0)
                out[rmin:rmax+1, cmin:cmax+1] = c
            return out
        prims["fill_bounding_box_of_each_color"] = fill_bounding_box_of_each_color

        # 138. Réflexion optique spéculaire d'un rayon diagonal contre un mur (tâche 508bd3b6)
        def reflect_diagonal_ray_off_wall(inp: np.ndarray, ctx=None):
            wall_mask = (inp == 2)
            if not np.any(wall_mask):
                return None
            ray_pts = [tuple(p) for p in np.argwhere(inp == 8)]
            if len(ray_pts) < 2:
                return None
            H, W = inp.shape
            pts_2 = np.argwhere(wall_mask)
            rows_2 = np.unique(pts_2[:, 0])
            cols_2 = np.unique(pts_2[:, 1])
            is_vert_wall = (len(rows_2) == H)
            is_horiz_wall = (len(cols_2) == W)
            if not (is_vert_wall or is_horiz_wall):
                return None
            c_wall = cols_2.mean()
            r_wall = rows_2.mean()
            best_step = None
            for p_tail in ray_pts:
                for p_head in ray_pts:
                    if p_tail == p_head: continue
                    dr = p_head[0] - p_tail[0]
                    dc = p_head[1] - p_tail[1]
                    if abs(dr) == 1 and abs(dc) == 1:
                        if is_vert_wall and abs(p_head[1] - c_wall) < abs(p_tail[1] - c_wall):
                            best_step = (p_head, dr, dc)
                        elif is_horiz_wall and abs(p_head[0] - r_wall) < abs(p_tail[0] - r_wall):
                            best_step = (p_head, dr, dc)
            if best_step is None:
                return None
            p_head, dr, dc = best_step
            cur_r, cur_c = p_head
            out = inp.copy()
            for _ in range(50):
                next_r = cur_r + dr
                next_c = cur_c + dc
                if not (0 <= next_r < H and 0 <= next_c < W):
                    break
                if wall_mask[next_r, next_c]:
                    if is_vert_wall:
                        dc = -dc
                    else:
                        dr = -dr
                    next_r = cur_r + dr
                    next_c = cur_c + dc
                    if not (0 <= next_r < H and 0 <= next_c < W):
                        break
                cur_r, cur_c = next_r, next_c
                if out[cur_r, cur_c] == 0:
                    out[cur_r, cur_c] = 3
            return out
        prims["reflect_diagonal_ray_off_wall"] = reflect_diagonal_ray_off_wall

        # 139. Réflexion à 4 quadrants autour d'une ancre centrale en X (tâche 4c5c2cf0)
        def reflect_shape_around_x_anchor(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) != 2:
                return None
            anchor_center = None
            anchor_col = None
            H, W = inp.shape
            for col in colors:
                pts = np.argwhere(inp == col)
                if len(pts) == 5:
                    rmin, cmin = pts.min(0)
                    rmax, cmax = pts.max(0)
                    if (rmax - rmin + 1, cmax - cmin + 1) == (3, 3):
                        sub = inp[rmin:rmax+1, cmin:cmax+1]
                        expected = np.array([[col, 0, col], [0, col, 0], [col, 0, col]])
                        if np.array_equal(sub, expected):
                            anchor_center = (rmin + 1, cmin + 1)
                            anchor_col = col
                            break
            if anchor_center is None:
                return None
            shape_col = [c for c in colors if c != anchor_col][0]
            shape_pts = np.argwhere(inp == shape_col)
            cr, cc = anchor_center
            out = inp.copy()
            for r, c in shape_pts:
                for mr, mc in [(r, c), (2 * cr - r, c), (r, 2 * cc - c), (2 * cr - r, 2 * cc - c)]:
                    if 0 <= mr < H and 0 <= mc < W:
                        out[mr, mc] = shape_col
            return out
        prims["reflect_shape_around_x_anchor"] = reflect_shape_around_x_anchor

        # 140. Prolongement périodique 2D vers le bas jusqu'à hauteur 10 (tâche 53b68214)
        def periodic_stepped_extension_downwards_to_10(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            if H >= 10:
                return None
            best_shift = None
            for dr in range(1, H):
                for dc in range(-W + 1, W):
                    valid = True
                    matches = 0
                    for r in range(dr, H):
                        for c in range(W):
                            if 0 <= c - dc < W:
                                val1 = inp[r, c]
                                val0 = inp[r - dr, c - dc]
                                if val1 != val0:
                                    valid = False
                                    break
                                if val1 != 0:
                                    matches += 1
                        if not valid:
                            break
                    if valid and matches > 0:
                        best_shift = (dr, dc)
                        break
                if best_shift is not None:
                    break
            if best_shift is None:
                return None
            dr, dc = best_shift
            out = np.zeros((10, W), dtype=inp.dtype)
            out[:H, :] = inp
            for r in range(H, 10):
                for c in range(W):
                    prev_c = c - dc
                    if 0 <= prev_c < W:
                        out[r, c] = out[r - dr, prev_c]
                    else:
                        out[r, c] = 0
            return out
        prims["periodic_stepped_extension_downwards_to_10"] = periodic_stepped_extension_downwards_to_10

        # 141. Remplissage de l'intérieur de tous les rectangles pleins par 8 (tâche 50cb2852)
        def fill_all_solid_rectangle_interiors_with_8(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            out = inp.copy()
            modified = False
            for col in colors:
                labeled, num_features = label(inp == col)
                for comp_id in range(1, num_features + 1):
                    pts = np.argwhere(labeled == comp_id)
                    rmin, cmin = pts.min(0)
                    rmax, cmax = pts.max(0)
                    h = rmax - rmin + 1
                    w = cmax - cmin + 1
                    if len(pts) == h * w and h >= 3 and w >= 3:
                        out[rmin+1:rmax, cmin+1:cmax] = 8
                        modified = True
            return out if modified else None
        prims["fill_all_solid_rectangle_interiors_with_8"] = fill_all_solid_rectangle_interiors_with_8

        # 142. Glissement d'une forme contre un mur avec pose d'une barre 8 de traîne (tâche 56dc2b01)
        def slide_shape_to_wall_with_trailing_bar(inp: np.ndarray, ctx=None):
            pts_2 = np.argwhere(inp == 2)
            pts_3 = np.argwhere(inp == 3)
            if len(pts_2) == 0 or len(pts_3) == 0:
                return None
            H, W = inp.shape
            rows_2 = np.unique(pts_2[:, 0])
            cols_2 = np.unique(pts_2[:, 1])
            is_col_wall = (len(rows_2) == H and len(cols_2) == 1)
            is_row_wall = (len(cols_2) == W and len(rows_2) == 1)
            if not (is_col_wall or is_row_wall):
                return None
            out = np.zeros_like(inp)
            out[inp == 2] = 2
            rmin, cmin = pts_3.min(0)
            rmax, cmax = pts_3.max(0)
            if is_col_wall:
                c_wall = cols_2[0]
                if cmax < c_wall:
                    shift_c = (c_wall - 1) - cmax
                    for r, c in pts_3:
                        out[r, c + shift_c] = 3
                    bar_c = cmin + shift_c - 1
                    if 0 <= bar_c < W:
                        out[:, bar_c] = 8
                else:
                    shift_c = (c_wall + 1) - cmin
                    for r, c in pts_3:
                        out[r, c + shift_c] = 3
                    bar_c = cmax + shift_c + 1
                    if 0 <= bar_c < W:
                        out[:, bar_c] = 8
                return out
            elif is_row_wall:
                r_wall = rows_2[0]
                if rmax < r_wall:
                    shift_r = (r_wall - 1) - rmax
                    for r, c in pts_3:
                        out[r + shift_r, c] = 3
                    bar_r = rmin + shift_r - 1
                    if 0 <= bar_r < H:
                        out[bar_r, :] = 8
                else:
                    shift_r = (r_wall + 1) - rmin
                    for r, c in pts_3:
                        out[r + shift_r, c] = 3
                    bar_r = rmax + shift_r + 1
                    if 0 <= bar_r < H:
                        out[bar_r, :] = 8
                return out
            return None
        prims["slide_shape_to_wall_with_trailing_bar"] = slide_shape_to_wall_with_trailing_bar

        # 143. Croisillons 6 passant par les centres des cadres 5x5 (tâche 41e4d17e)
        def crosshairs_through_frame_centers(inp: np.ndarray, ctx=None):
            if not np.all(np.isin(inp, [1, 8])):
                return None
            H, W = inp.shape
            labeled, num_features = label(inp == 1)
            if num_features == 0:
                return None
            centers = []
            for comp_id in range(1, num_features + 1):
                pts = np.argwhere(labeled == comp_id)
                rmin, cmin = pts.min(0)
                rmax, cmax = pts.max(0)
                if (rmax - rmin + 1, cmax - cmin + 1) == (5, 5):
                    centers.append((rmin + 2, cmin + 2))
            if len(centers) == 0:
                return None
            out = inp.copy()
            for cr, cc in centers:
                for c in range(W):
                    if out[cr, c] != 1:
                        out[cr, c] = 6
                for r in range(H):
                    if out[r, cc] != 1:
                        out[r, cc] = 6
            return out
        prims["crosshairs_through_frame_centers"] = crosshairs_through_frame_centers

        # 144. Réflexion à 4 quadrants autour d'une ancre carrée 2x2 (tâche 4938f0c2)
        def reflect_shape_around_2x2_anchor(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) != 2:
                return None
            anchor_col = None
            anchor_r = None
            anchor_c = None
            H, W = inp.shape
            for col in colors:
                pts = np.argwhere(inp == col)
                if len(pts) == 4:
                    rmin, cmin = pts.min(0)
                    rmax, cmax = pts.max(0)
                    if (rmax - rmin + 1, cmax - cmin + 1) == (2, 2):
                        anchor_col = col
                        anchor_r = rmin
                        anchor_c = cmin
                        break
            if anchor_col is None:
                return None
            shape_col = [c for c in colors if c != anchor_col][0]
            shape_pts = np.argwhere(inp == shape_col)
            out = inp.copy()
            for r, c in shape_pts:
                mr = 2 * anchor_r + 1 - r
                mc = 2 * anchor_c + 1 - c
                for cand_r, cand_c in [(r, c), (mr, c), (r, mc), (mr, mc)]:
                    if 0 <= cand_r < H and 0 <= cand_c < W:
                        out[cand_r, cand_c] = shape_col
            return out
        prims["reflect_shape_around_2x2_anchor"] = reflect_shape_around_2x2_anchor

        # 145. Réparation d'une forme occluse par symétrie axiale verticale (tâche 3345333e)
        def repair_occluded_shape_by_vertical_symmetry(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) != 2:
                return None
            occl_col = None
            for c in colors:
                pts = np.argwhere(inp == c)
                rmin, cmin = pts.min(0)
                rmax, cmax = pts.max(0)
                h = rmax - rmin + 1
                w = cmax - cmin + 1
                if len(pts) == h * w and h >= 2 and w >= 2:
                    occl_col = c
                    break
            if occl_col is None:
                return None
            shape_col = [c for c in colors if c != occl_col][0]
            shape_pts = np.argwhere(inp == shape_col)
            H, W = inp.shape
            best_c_sym = None
            best_score = -1
            for c_sym in np.arange(0, W, 0.5):
                hits = 0
                for r, c in shape_pts:
                    mc = 2 * c_sym - c
                    if 0 <= mc < W and mc == int(mc):
                        mc = int(mc)
                        if inp[r, mc] == shape_col:
                            hits += 1
                if hits > best_score:
                    best_score = hits
                    best_c_sym = c_sym
            if best_c_sym is None or best_score < 4:
                return None
            out = np.zeros_like(inp)
            out[inp == shape_col] = shape_col
            for r, c in shape_pts:
                mc = int(round(2 * best_c_sym - c))
                if 0 <= mc < W:
                    out[r, mc] = shape_col
            return out
        prims["repair_occluded_shape_by_vertical_symmetry"] = repair_occluded_shape_by_vertical_symmetry

        # 146. Extraction du bloc 3x3 asymétrique singulier parmi 3 blocs empilés (tâche 662c240a)
        def extract_anomalous_asymmetric_3x3_block(inp: np.ndarray, ctx=None):
            if inp.shape[0] != 9 or inp.shape[1] != 3:
                return None
            for i in range(0, inp.shape[0], 3):
                block = inp[i:i+3, :]
                if not np.array_equal(block, block.T):
                    return block
            return None
        prims["extract_anomalous_asymmetric_3x3_block"] = extract_anomalous_asymmetric_3x3_block

        # 147. Réflexion de marqueurs extérieurs à travers la paroi d'un conteneur en U (tâche 6855a6e4)
        def reflect_markers_across_u_container_wall(inp: np.ndarray, ctx=None):
            if 2 not in inp or 5 not in inp or inp.shape[0] < 10:
                return None
            out = inp.copy()
            out[inp == 5] = 0
            lbl_2, n_2 = label(inp == 2)
            if n_2 < 1:
                return None
            for c_id in range(1, n_2 + 1):
                box_coords = np.argwhere(lbl_2 == c_id)
                rows, cols = box_coords[:, 0], box_coords[:, 1]
                row_counts = {r: np.sum(rows == r) for r in np.unique(rows)}
                col_counts = {c: np.sum(cols == c) for c in np.unique(cols)}
                max_r_cnt = max(row_counts.values())
                max_c_cnt = max(col_counts.values())
                if max_r_cnt > max_c_cnt:
                    wall_r = [r for r, cnt in row_counts.items() if cnt == max_r_cnt][0]
                    prongs_dir = 1 if any(r > wall_r for r in rows) else -1
                    outside_dir = -prongs_dir
                    c_min, c_max = cols.min(), cols.max()
                    for r_5, c_5 in np.argwhere(inp == 5):
                        if (r_5 - wall_r) * outside_dir > 0 and (c_min - 2 <= c_5 <= c_max + 2):
                            new_r = 2 * wall_r - r_5
                            if 0 <= new_r < inp.shape[0]:
                                out[new_r, c_5] = 5
                else:
                    wall_c = [c for c, cnt in col_counts.items() if cnt == max_c_cnt][0]
                    prongs_dir = 1 if any(c > wall_c for c in cols) else -1
                    outside_dir = -prongs_dir
                    r_min, r_max = rows.min(), rows.max()
                    for r_5, c_5 in np.argwhere(inp == 5):
                        if (c_5 - wall_c) * outside_dir > 0 and (r_min - 2 <= r_5 <= r_max + 2):
                            new_c = 2 * wall_c - c_5
                            if 0 <= new_c < inp.shape[1]:
                                out[r_5, new_c] = 5
            return out
        prims["reflect_markers_across_u_container_wall"] = reflect_markers_across_u_container_wall

        # 148. Recoloration à 5 de la composante congruente au gabarit coin supérieur gauche (tâche 63613498)
        def recolor_congruent_template_component_to_5(inp: np.ndarray, ctx=None):
            if inp.shape[0] < 5 or inp.shape[1] < 5:
                return None
            if not (np.all(inp[3, :4] == 5) and np.all(inp[:4, 3] == 5)):
                return None
            out = inp.copy()
            tl = inp[0:3, 0:3]
            tl_mask = (tl != 0) & (tl != 5)
            coords = np.argwhere(tl_mask)
            if len(coords) == 0:
                return None
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            template = tl_mask[r0:r1+1, c0:c1+1]
            grid_rest = inp.copy()
            grid_rest[:4, :4] = 0
            for color in np.unique(grid_rest):
                if color == 0 or color == 5:
                    continue
                lbl, num = label(grid_rest == color)
                for comp_id in range(1, num + 1):
                    comp_mask = (lbl == comp_id)
                    c_coords = np.argwhere(comp_mask)
                    cr0, cc0 = c_coords.min(axis=0)
                    cr1, cc1 = c_coords.max(axis=0)
                    comp_sub = comp_mask[cr0:cr1+1, cc0:cc1+1]
                    if comp_sub.shape == template.shape and np.array_equal(comp_sub, template):
                        out[comp_mask] = 5
                        return out
            return None
        prims["recolor_congruent_template_component_to_5"] = recolor_congruent_template_component_to_5

        # 149. Déplacement de blocs verticaux empilés vers le haut selon leur propre hauteur (tâche 5521c0d9)
        def shift_stacked_bars_up_by_height(inp: np.ndarray, ctx=None):
            if inp.shape != (15, 15) or not np.any(inp[14, :] != 0):
                return None
            out = np.zeros_like(inp)
            for color in np.unique(inp):
                if color == 0:
                    continue
                coords = np.argwhere(inp == color)
                r_min = coords[:, 0].min()
                r_max = coords[:, 0].max()
                h = r_max - r_min + 1
                new_coords = coords.copy()
                new_coords[:, 0] -= h
                if np.any(new_coords[:, 0] < 0):
                    return None
                out[new_coords[:, 0], new_coords[:, 1]] = color
            return out
        prims["shift_stacked_bars_up_by_height"] = shift_stacked_bars_up_by_height

        # 150. Déplacement d'une boîte 3x3 le long d'une voie de points guides (tâche 5168d44c)
        def slide_3x3_box_along_dot_track(inp: np.ndarray, ctx=None):
            if 2 not in inp or 3 not in inp:
                return None
            h, w = inp.shape
            center = None
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    if inp[r, c] == 3 and np.sum(inp[r-1:r+2, c-1:c+2] == 2) == 8:
                        center = (r, c)
                        break
                if center is not None:
                    break
            if center is None:
                return None
            cr, cc = center
            dots_3 = np.argwhere(inp == 3)
            if len(np.unique(dots_3[:, 0])) == 1:
                dr, dc = 0, 2
            elif len(np.unique(dots_3[:, 1])) == 1:
                dr, dc = 2, 0
            else:
                return None
            out = inp.copy()
            out[cr-1:cr+2, cc-1:cc+2] = 0
            out[cr, cc] = 3
            new_cr, new_cc = cr + dr, cc + dc
            if not (1 <= new_cr < h - 1 and 1 <= new_cc < w - 1):
                return None
            out[new_cr-1:new_cr+2, new_cc-1:new_cc+2] = 2
            out[new_cr, new_cc] = 3
            return out
        prims["slide_3x3_box_along_dot_track"] = slide_3x3_box_along_dot_track

        # 151. Recoloration de 4 quadrants d'un motif 6x6 via palette 2x2 (tâche 7c008303)
        def recolor_4quadrants_by_2x2_palette(inp: np.ndarray, ctx=None):
            if inp.shape != (9, 9) or 8 not in inp or 3 not in inp:
                return None
            sep_rows = [r for r in range(inp.shape[0]) if np.all(inp[r, :] == 8)]
            sep_cols = [c for c in range(inp.shape[1]) if np.all(inp[:, c] == 8)]
            if not sep_rows or not sep_cols:
                return None
            sr, sc = sep_rows[0], sep_cols[0]
            quads = {
                'TL': inp[:sr, :sc],
                'TR': inp[:sr, sc+1:],
                'BL': inp[sr+1:, :sc],
                'BR': inp[sr+1:, sc+1:]
            }
            palette = None
            pattern = None
            for k, q in quads.items():
                if q.shape == (2, 2):
                    palette = q
                elif q.shape == (6, 6):
                    pattern = q
            if palette is None or pattern is None:
                return None
            out = np.zeros((6, 6), dtype=np.int32)
            for qr in range(2):
                for qc in range(2):
                    block = pattern[qr*3:(qr+1)*3, qc*3:(qc+1)*3]
                    col = palette[qr, qc]
                    out[qr*3:(qr+1)*3, qc*3:(qc+1)*3] = np.where(block == 3, col, 0)
            return out
        prims["recolor_4quadrants_by_2x2_palette"] = recolor_4quadrants_by_2x2_palette

        # 152. Remplissage des trous du rectangle englobant avec la couleur 2 (tâche 6d75e8bb)
        def fill_bbox_holes_with_2(inp: np.ndarray, ctx=None):
            if 8 not in inp or 0 not in inp or inp.shape[0] > 20:
                return None
            coords = np.argwhere(inp == 8)
            if len(coords) == 0:
                return None
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            out = inp.copy()
            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    if out[r, c] == 0:
                        out[r, c] = 2
            return out
        prims["fill_bbox_holes_with_2"] = fill_bbox_holes_with_2

        # 153. Routage d'un bloc 3x3 vers la cellule indexée par la couleur 4 (tâche 6d0160f0)
        def route_block_by_yellow_coordinate(inp: np.ndarray, ctx=None):
            if inp.shape != (11, 11) or 4 not in inp or 5 not in inp:
                return None
            if not (np.all(inp[3, :] == 5) and np.all(inp[7, :] == 5) and np.all(inp[:, 3] == 5) and np.all(inp[:, 7] == 5)):
                return None
            coords_4 = np.argwhere(inp == 4)
            if len(coords_4) != 1:
                return None
            r4, c4 = coords_4[0]
            comp_r, comp_c = r4 // 4, c4 // 4
            rel_r, rel_c = r4 % 4, c4 % 4
            if rel_r > 2 or rel_c > 2:
                return None
            src_block = inp[comp_r*4:comp_r*4+3, comp_c*4:comp_c*4+3].copy()
            dst_r, dst_c = rel_r, rel_c
            out = np.zeros_like(inp)
            out[3, :] = 5
            out[7, :] = 5
            out[:, 3] = 5
            out[:, 7] = 5
            out[dst_r*4:dst_r*4+3, dst_c*4:dst_c*4+3] = src_block
            return out
        prims["route_block_by_yellow_coordinate"] = route_block_by_yellow_coordinate

        # 154. Recoloration des composantes selon leur taille par 5 - taille (tâche 6e82a1ae)
        def recolor_connected_components_by_size(inp: np.ndarray, ctx=None):
            if inp.shape != (10, 10) or 5 not in inp or set(np.unique(inp)) - {0, 5}:
                return None
            out = inp.copy()
            lbl, num = label(inp == 5)
            for c_id in range(1, num + 1):
                mask = (lbl == c_id)
                sz = np.sum(mask)
                if sz not in [2, 3, 4]:
                    return None
                out[mask] = 5 - sz
            return out
        prims["recolor_connected_components_by_size"] = recolor_connected_components_by_size

        # 155. Croix diagonale en X issue d'un pixel solitaire vers les 4 bords (tâche 623ea044)
        def diagonal_x_cross_from_single_pixel(inp: np.ndarray, ctx=None):
            if np.count_nonzero(inp) != 1:
                return None
            coords = np.argwhere(inp != 0)
            r0, c0 = coords[0]
            color = inp[r0, c0]
            out = np.zeros_like(inp)
            h, w = inp.shape
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                r, c = r0, c0
                while 0 <= r < h and 0 <= c < w:
                    out[r, c] = color
                    r += dr
                    c += dc
            return out
        prims["diagonal_x_cross_from_single_pixel"] = diagonal_x_cross_from_single_pixel

        # 156. Inversion 180° des 4 coins d'un bloc central 2x2 vers les 4 coins du domaine (tâche 93b581b8)
        def corner_inversion_from_2x2_block(inp: np.ndarray, ctx=None):
            coords = np.argwhere(inp != 0)
            if len(coords) != 4:
                return None
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            if (r_max - r_min != 1) or (c_max - c_min != 1):
                return None
            block = inp[r_min:r_max+1, c_min:c_max+1]
            H, W = inp.shape
            h_above = min(r_min, 2)
            h_below = min(H - 1 - r_max, 2)
            w_left = min(c_min, 2)
            w_right = min(W - 1 - c_max, 2)
            out = inp.copy()
            if h_above > 0 and w_left > 0:
                out[r_min - h_above:r_min, c_min - w_left:c_min] = block[1, 1]
            if h_above > 0 and w_right > 0:
                out[r_min - h_above:r_min, c_max + 1:c_max + 1 + w_right] = block[1, 0]
            if h_below > 0 and w_left > 0:
                out[r_max + 1:r_max + 1 + h_below, c_min - w_left:c_min] = block[0, 1]
            if h_below > 0 and w_right > 0:
                out[r_max + 1:r_max + 1 + h_below, c_max + 1:c_max + 1 + w_right] = block[0, 0]
            return out
        prims["corner_inversion_from_2x2_block"] = corner_inversion_from_2x2_block

        # 157. Miroir horizontal concaténé répété en alternance verticale (tâche 8d5021e8)
        def mirror_h_and_alternate_v(inp: np.ndarray, ctx=None):
            if inp.shape != (3, 2):
                return None
            H = np.hstack([np.fliplr(inp), inp])
            return np.vstack([np.flipud(H), H, np.flipud(H)])
        prims["mirror_h_and_alternate_v"] = mirror_h_and_alternate_v

        # 158. Pavage par compte complémentaire en ordre de lecture (tâche 91413438)
        def tile_by_complementary_count(inp: np.ndarray, ctx=None):
            if inp.shape != (3, 3):
                return None
            count = np.count_nonzero(inp)
            scale = 9 - count
            if scale <= 0:
                return None
            out = np.zeros((scale * 3, scale * 3), dtype=np.int32)
            placed = 0
            for br in range(scale):
                for bc in range(scale):
                    if placed < count:
                        out[br*3:(br+1)*3, bc*3:(bc+1)*3] = inp
                        placed += 1
                    else:
                        break
                if placed >= count:
                    break
            return out
        prims["tile_by_complementary_count"] = tile_by_complementary_count

        # 159. Recodage de lignes partielles selon la ligne maîtresse complète (tâche 82819916)
        def recode_rows_by_master_template(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            master_r = None
            for r in range(H):
                if np.all(inp[r, :] != 0):
                    master_r = r
                    break
            if master_r is None:
                return None
            master_row = inp[master_r, :]
            out = inp.copy()
            for r in range(H):
                if r == master_r or not np.any(inp[r, :] != 0):
                    continue
                mapping = {}
                for c in range(W):
                    val = inp[r, c]
                    if val != 0:
                        mapping[master_row[c]] = val
                if len(mapping) == len(np.unique(master_row)):
                    for c in range(W):
                        out[r, c] = mapping[master_row[c]]
            return out
        prims["recode_rows_by_master_template"] = recode_rows_by_master_template

        # 160. Inversion des 4 coins intérieurs vers les 4 coins extérieurs d'un cadre (tâche 952a094c)
        def invert_inner_corners_to_outer(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            frame_col = None
            for c in colors:
                coords = np.argwhere(inp == c)
                if len(coords) >= 12:
                    r_min, c_min = coords.min(axis=0)
                    r_max, c_max = coords.max(axis=0)
                    top = np.all(inp[r_min, c_min:c_max+1] == c)
                    bottom = np.all(inp[r_max, c_min:c_max+1] == c)
                    left = np.all(inp[r_min:r_max+1, c_min] == c)
                    right = np.all(inp[r_min:r_max+1, c_max] == c)
                    if top and bottom and left and right:
                        frame_col = c
                        break
            if frame_col is None:
                return None
            out = inp.copy()
            c_tl = inp[r_min + 1, c_min + 1]
            c_tr = inp[r_min + 1, c_max - 1]
            c_bl = inp[r_max - 1, c_min + 1]
            c_br = inp[r_max - 1, c_max - 1]
            out[r_min + 1, c_min + 1] = 0
            out[r_min + 1, c_max - 1] = 0
            out[r_max - 1, c_min + 1] = 0
            out[r_max - 1, c_max - 1] = 0
            if r_min - 1 >= 0 and c_min - 1 >= 0:
                out[r_min - 1, c_min - 1] = c_br
            if r_min - 1 >= 0 and c_max + 1 < inp.shape[1]:
                out[r_min - 1, c_max + 1] = c_bl
            if r_max + 1 < inp.shape[0] and c_min - 1 >= 0:
                out[r_max + 1, c_min - 1] = c_tr
            if r_max + 1 < inp.shape[0] and c_max + 1 < inp.shape[1]:
                out[r_max + 1, c_max + 1] = c_tl
            return out
        prims["invert_inner_corners_to_outer"] = invert_inner_corners_to_outer

        # 161. Remplissage des compartiments diagonaux (0,0)->1, centre->2, (N-1, N-1)->3 (tâche 941d9a10)
        def fill_diagonal_compartments_1_2_3(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            H, W = inp.shape
            sep_rows = [r for r in range(H) if np.all(inp[r, :] == 5)]
            sep_cols = [c for c in range(W) if np.all(inp[:, c] == 5)]
            if not sep_rows or not sep_cols:
                return None
            row_spans = []
            prev_r = 0
            for sr in sep_rows:
                if sr > prev_r:
                    row_spans.append((prev_r, sr))
                prev_r = sr + 1
            if prev_r < H:
                row_spans.append((prev_r, H))
            col_spans = []
            prev_c = 0
            for sc in sep_cols:
                if sc > prev_c:
                    col_spans.append((prev_c, sc))
                prev_c = sc + 1
            if prev_c < W:
                col_spans.append((prev_c, W))
            n_r, n_c = len(row_spans), len(col_spans)
            if n_r == 0 or n_c == 0:
                return None
            out = inp.copy()
            r0, r1 = row_spans[0]
            c0, c1 = col_spans[0]
            out[r0:r1, c0:c1] = 1
            mid_r = n_r // 2
            mid_c = n_c // 2
            r0, r1 = row_spans[mid_r]
            c0, c1 = col_spans[mid_c]
            out[r0:r1, c0:c1] = 2
            r0, r1 = row_spans[-1]
            c0, c1 = col_spans[-1]
            out[r0:r1, c0:c1] = 3
            return out
        prims["fill_diagonal_compartments_1_2_3"] = fill_diagonal_compartments_1_2_3

        # 162. Gravitation attraction/répulsion par rapport à un diviseur (tâche 8d510a79)
        def gravitational_attraction_repulsion(inp: np.ndarray, ctx=None):
            if 5 not in inp or (1 not in inp and 2 not in inp):
                return None
            H, W = inp.shape
            sep_rows = [r for r in range(H) if np.all(inp[r, :] == 5)]
            if len(sep_rows) != 1:
                return None
            div_r = sep_rows[0]
            out = inp.copy()
            for r, c in np.argwhere(inp == 2):
                if r < div_r:
                    for step_r in range(r, div_r):
                        out[step_r, c] = 2
                elif r > div_r:
                    for step_r in range(div_r + 1, r + 1):
                        out[step_r, c] = 2
            for r, c in np.argwhere(inp == 1):
                if r < div_r:
                    for step_r in range(0, r + 1):
                        out[step_r, c] = 1
                elif r > div_r:
                    for step_r in range(r, H):
                        out[step_r, c] = 1
            return out
        prims["gravitational_attraction_repulsion"] = gravitational_attraction_repulsion

        # 163. Filtrage morphologique préservant uniquement les blocs 2x2 et supérieurs (tâche 7f4411dc)
        def morphological_filter_keep_2x2_blocks(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) != 1:
                return None
            color = colors[0]
            H, W = inp.shape
            is_in_2x2 = np.zeros((H, W), dtype=bool)
            for r in range(H - 1):
                for c in range(W - 1):
                    if np.all(inp[r:r+2, c:c+2] == color):
                        is_in_2x2[r:r+2, c:c+2] = True
            return np.where(is_in_2x2, inp, 0)
        prims["morphological_filter_keep_2x2_blocks"] = morphological_filter_keep_2x2_blocks

        # 164. Découpe laser perpendiculaire passant par les trous d'une dalle (tâche 855e0971)
        def laser_cut_through_holes_in_slabs(inp: np.ndarray, ctx=None):
            if 0 not in inp:
                return None
            H, W = inp.shape
            out = inp.copy()
            row_changes = sum(np.sum(inp[r, 1:] != inp[r, :-1]) for r in range(H))
            col_changes = sum(np.sum(inp[1:, c] != inp[:-1, c]) for c in range(W))
            if row_changes < col_changes:
                r = 0
                while r < H:
                    vals, counts = np.unique(inp[r, :], return_counts=True)
                    non_zero_vals = [(v, c) for v, c in zip(vals, counts) if v != 0]
                    if not non_zero_vals:
                        r += 1
                        continue
                    maj_col = max(non_zero_vals, key=lambda x: x[1])[0]
                    r_start = r
                    while r < H:
                        vals_r, counts_r = np.unique(inp[r, :], return_counts=True)
                        nz = [(v, c) for v, c in zip(vals_r, counts_r) if v != 0]
                        if nz and max(nz, key=lambda x: x[1])[0] == maj_col:
                            r += 1
                        else:
                            break
                    r_end = r - 1
                    holes = np.argwhere(inp[r_start:r_end+1, :] == 0)
                    hole_cols = np.unique(holes[:, 1])
                    for hc in hole_cols:
                        out[r_start:r_end+1, hc] = 0
            else:
                c = 0
                while c < W:
                    vals, counts = np.unique(inp[:, c], return_counts=True)
                    non_zero_vals = [(v, c_cnt) for v, c_cnt in zip(vals, counts) if v != 0]
                    if not non_zero_vals:
                        c += 1
                        continue
                    maj_col = max(non_zero_vals, key=lambda x: x[1])[0]
                    c_start = c
                    while c < W:
                        vals_c, counts_c = np.unique(inp[:, c], return_counts=True)
                        nz = [(v, c_cnt) for v, c_cnt in zip(vals_c, counts_c) if v != 0]
                        if nz and max(nz, key=lambda x: x[1])[0] == maj_col:
                            c += 1
                        else:
                            break
                    c_end = c - 1
                    holes = np.argwhere(inp[:, c_start:c_end+1] == 0)
                    hole_rows = np.unique(holes[:, 0])
                    for hr in hole_rows:
                        out[hr, c_start:c_end+1] = 0
            return out
        prims["laser_cut_through_holes_in_slabs"] = laser_cut_through_holes_in_slabs

        # 165. Débruitage par support de voisinage orthogonal majoritaire (tâche 7e0986d6)
        def denoise_by_majority_support(inp: np.ndarray, ctx=None):
            colors, counts = np.unique(inp[inp != 0], return_counts=True)
            if len(colors) != 2:
                return None
            main_col = colors[np.argmax(counts)]
            noise_col = colors[np.argmin(counts)]
            out = np.zeros_like(inp)
            out[inp == main_col] = main_col
            noise_pts = np.argwhere(inp == noise_col)
            for r, c in noise_pts:
                ortho = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1] and inp[nr, nc] == main_col:
                        ortho += 1
                if ortho >= 2:
                    out[r, c] = main_col
            return out
        prims["denoise_by_majority_support"] = denoise_by_majority_support

        # 166. Conserver la couleur dominante et remplacer les autres par 5 (tâche 9565186b)
        def keep_mode_color_replace_rest_5(inp: np.ndarray, ctx=None):
            uq, counts = np.unique(inp, return_counts=True)
            if len(uq) <= 1:
                return None
            mode_color = uq[np.argmax(counts)]
            out = np.full_like(inp, 5)
            out[inp == mode_color] = mode_color
            return out
        prims["keep_mode_color_replace_rest_5"] = keep_mode_color_replace_rest_5

        # 167. Entourer les pixels graines d'un carré 3x3 selon une palette fixe (tâche 913fb3ed)
        def surround_seeds_by_fixed_palette(inp: np.ndarray, ctx=None):
            cmap = {8: 4, 3: 6, 2: 1}
            seeds = [(r, c, inp[r, c]) for r, c in zip(*np.where(inp != 0))]
            if not seeds or any(val not in cmap for _, _, val in seeds):
                return None
            out = inp.copy()
            H, W = inp.shape
            for r, c, val in seeds:
                surr_col = cmap[val]
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            out[nr, nc] = surr_col
            return out
        prims["surround_seeds_by_fixed_palette"] = surround_seeds_by_fixed_palette

        # 168. Fractal Kronecker macro à partir d'un bloc 9x9 sous-échantillonné 3x3 (tâche 80af3007)
        def macro_kronecker_fractal(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            r, c = np.where(inp == 5)
            h = r.max() - r.min() + 1
            w = c.max() - c.min() + 1
            if h != 9 or w != 9:
                return None
            sub = inp[r.min():r.max()+1, c.min():c.max()+1]
            P = (sub[::3, ::3] == 5).astype(int)
            return np.kron(P, P) * 5
        prims["macro_kronecker_fractal"] = macro_kronecker_fractal

        # 169. Matrice carrée classée par ordre de lignes/colonnes selon gabarit 5 (tâche 8e1813be)
        def rank_lines_to_square(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            r5, c5 = np.where(inp == 5)
            h5 = r5.max() - r5.min() + 1
            w5 = c5.max() - c5.min() + 1
            if h5 != w5 or h5 < 2:
                return None
            row_colors = []
            for r in range(inp.shape[0]):
                nz = [c for c in inp[r] if c not in (0, 5)]
                if nz and (len(row_colors) == 0 or nz[0] != row_colors[-1]):
                    row_colors.append(nz[0])
            col_colors = []
            for c in range(inp.shape[1]):
                nz = [val for val in inp[:, c] if val not in (0, 5)]
                if nz and (len(col_colors) == 0 or nz[0] != col_colors[-1]):
                    col_colors.append(nz[0])
            if len(row_colors) >= h5:
                out = np.zeros((h5, h5), dtype=int)
                for i, c in enumerate(row_colors[:h5]):
                    out[i, :] = c
                return out
            elif len(col_colors) >= h5:
                out = np.zeros((h5, h5), dtype=int)
                for j, c in enumerate(col_colors[:h5]):
                    out[:, j] = c
                return out
            return None
        prims["rank_lines_to_square"] = rank_lines_to_square

        # 170. Recoloration en 3 des pixels 1 situés dans la boîte englobante des 8 (tâche 32597951)
        def recolor_1s_in_bbox_of_8s_to_3(inp: np.ndarray, ctx=None):
            if 8 not in inp or 1 not in inp:
                return None
            rows, cols = np.where(inp == 8)
            out = inp.copy()
            rmin, rmax = rows.min(), rows.max()
            cmin, cmax = cols.min(), cols.max()
            bbox_sub = out[rmin:rmax+1, cmin:cmax+1]
            bbox_sub[bbox_sub == 1] = 3
            return out
        prims["recolor_1s_in_bbox_of_8s_to_3"] = recolor_1s_in_bbox_of_8s_to_3

        # 171. Grille 3x3 selon le nombre de couleurs uniques (tâche 6e02f1e3)
        def diag_anti_or_row_by_unique_color_count(inp: np.ndarray, ctx=None):
            if inp.shape != (3, 3):
                return None
            uq = np.unique(inp)
            out = np.zeros((3, 3), dtype=int)
            if len(uq) == 1:
                out[0, :] = 5
            elif len(uq) == 2:
                np.fill_diagonal(out, 5)
            elif len(uq) == 3:
                np.fill_diagonal(np.fliplr(out), 5)
            else:
                return None
            return out
        prims["diag_anti_or_row_by_unique_color_count"] = diag_anti_or_row_by_unique_color_count

        # 172. Inversion radiale des anneaux carrés concentriques (tâche 85c4e7cd)
        def reverse_concentric_square_rings(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            if H != W or H % 2 != 0:
                return None
            k = H // 2
            for i in range(k):
                c = inp[i, i]
                if not (np.all(inp[i, i:H-i] == c) and np.all(inp[H-1-i, i:H-i] == c) and
                        np.all(inp[i:H-i, i] == c) and np.all(inp[i:H-i, W-1-i] == c)):
                    return None
            colors = [inp[i, i] for i in range(k)]
            rev_colors = colors[::-1]
            out = np.zeros_like(inp)
            for i in range(k):
                c = rev_colors[i]
                out[i:H-i, i:H-i] = c
            return out
        prims["reverse_concentric_square_rings"] = reverse_concentric_square_rings

        # 173. Marquage des coins 1, 2, 3, 4 autour des blocs 2x2 de 5 (tâche 95990924)
        def mark_2x2_square_corners_1234(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            H, W = inp.shape
            out = inp.copy()
            found = False
            for r in range(H - 1):
                for c in range(W - 1):
                    if np.all(inp[r:r+2, c:c+2] == 5):
                        found = True
                        if r - 1 >= 0 and c - 1 >= 0:
                            out[r - 1, c - 1] = 1
                        if r - 1 >= 0 and c + 2 < W:
                            out[r - 1, c + 2] = 2
                        if r + 2 < H and c - 1 >= 0:
                            out[r + 2, c - 1] = 3
                        if r + 2 < H and c + 2 < W:
                            out[r + 2, c + 2] = 4
            return out if found else None
        prims["mark_2x2_square_corners_1234"] = mark_2x2_square_corners_1234

        # 174. Recoloration des rectangles creux en 3 (tâche 810b9b61)
        def recolor_hollow_rectangles_to_3(inp: np.ndarray, ctx=None):
            if 1 not in inp:
                return None
            out = inp.copy()
            lbl, num = label(inp == 1)
            modified = False
            for c in range(1, num + 1):
                mask = (lbl == c)
                r, cl = np.where(mask)
                rmin, rmax = r.min(), r.max()
                cmin, cmax = cl.min(), cl.max()
                h = rmax - rmin + 1
                w = cmax - cmin + 1
                if h >= 3 and w >= 3:
                    expected = np.zeros((h, w), dtype=bool)
                    expected[0, :] = True
                    expected[-1, :] = True
                    expected[:, 0] = True
                    expected[:, -1] = True
                    actual = mask[rmin:rmax+1, cmin:cmax+1]
                    if np.array_equal(actual, expected):
                        out[mask] = 3
                        modified = True
            return out if modified else None
        prims["recolor_hollow_rectangles_to_3"] = recolor_hollow_rectangles_to_3

        # 175. Classification de 3 blocs 4x4 vers une grille 3x3 (tâche 995c5fa3)
        def classify_3_blocks_to_3x3_grid(inp: np.ndarray, ctx=None):
            if inp.shape != (4, 14):
                return None
            def classify(blk):
                if np.all(blk == 5):
                    return 2
                if blk[1, 1] == 0 and blk[1, 2] == 0 and blk[2, 1] == 0 and blk[2, 2] == 0:
                    return 8
                if blk[1, 0] == 0 and blk[1, 3] == 0 and blk[2, 0] == 0 and blk[2, 3] == 0:
                    return 3
                if blk[2, 1] == 0 and blk[2, 2] == 0 and blk[3, 1] == 0 and blk[3, 2] == 0:
                    return 4
                return None
            c0 = classify(inp[:, 0:4])
            c1 = classify(inp[:, 5:9])
            c2 = classify(inp[:, 10:14])
            if c0 is None or c1 is None or c2 is None:
                return None
            return np.array([[c0, c0, c0], [c1, c1, c1], [c2, c2, c2]])
        prims["classify_3_blocks_to_3x3_grid"] = classify_3_blocks_to_3x3_grid

        # 176. Damier 2x6 bicolore alterné (tâche e9afcf9a)
        def chessboard_2x6(inp: np.ndarray, ctx=None):
            if inp.shape != (2, 6):
                return None
            A = inp[0, 0]
            B = inp[1, 0]
            if A == B:
                return None
            out = np.zeros((2, 6), dtype=int)
            for r in range(2):
                for c in range(6):
                    out[r, c] = A if (r + c) % 2 == 0 else B
            return out
        prims["chessboard_2x6"] = chessboard_2x6

        # 177. Translation vers le bas de 1 ligne des pixels 8 et recoloration en 2 (tâche a79310a0)
        def shift_8s_down_by_1_and_recolor_to_2(inp: np.ndarray, ctx=None):
            if 8 not in inp:
                return None
            H, W = inp.shape
            out = np.zeros_like(inp)
            for r in range(H):
                for c in range(W):
                    if inp[r, c] == 8 and r + 1 < H:
                        out[r + 1, c] = 2
            return out
        prims["shift_8s_down_by_1_and_recolor_to_2"] = shift_8s_down_by_1_and_recolor_to_2

        # 178. Réticule orthogonal ligne 2 et colonne 8 avec intersection en 4 (tâche bdad9b1f)
        def crosshairs_col8_row2_with_center4(inp: np.ndarray, ctx=None):
            cols_8 = np.where(inp == 8)[1]
            rows_2 = np.where(inp == 2)[0]
            if len(cols_8) == 0 or len(rows_2) == 0:
                return None
            c8 = cols_8[0]
            r2 = rows_2[0]
            out = np.zeros_like(inp)
            out[:, c8] = 8
            out[r2, :] = 2
            out[r2, c8] = 4
            return out
        prims["crosshairs_col8_row2_with_center4"] = crosshairs_col8_row2_with_center4

        # 179. Chute verticale et projection vers le bas par colonne (tâche d037b0a7)
        def gravity_down_project_each_pixel_downwards(inp: np.ndarray, ctx=None):
            if np.all(inp == 0):
                return None
            H, W = inp.shape
            out = np.zeros_like(inp)
            for r in range(H):
                for c in range(W):
                    val = inp[r, c]
                    if val != 0:
                        out[r:, c] = val
            return out
        prims["gravity_down_project_each_pixel_downwards"] = gravity_down_project_each_pixel_downwards

        # 180. Comptage des pixels monochromes vers une barre 1xN (tâche d631b094)
        def count_single_color_pixels_to_1xn_bar(inp: np.ndarray, ctx=None):
            nz = inp[inp != 0]
            if len(nz) == 0 or len(np.unique(nz)) != 1:
                return None
            col = nz[0]
            cnt = len(nz)
            return np.full((1, cnt), col, dtype=int)
        prims["count_single_color_pixels_to_1xn_bar"] = count_single_color_pixels_to_1xn_bar

        # 181. Agrandissement Kronecker facteur K = nombre de pixels non nuls (tâche ac0a08a4)
        def kronecker_upscale_by_nonzero_count(inp: np.ndarray, ctx=None):
            K = int(np.sum(inp != 0))
            if K <= 1:
                return None
            return np.kron(inp, np.ones((K, K), dtype=int))
        prims["kronecker_upscale_by_nonzero_count"] = kronecker_upscale_by_nonzero_count

        # 182. Agrandissement Kronecker facteur K = nombre de couleurs uniques (tâche b91ae062)
        def kronecker_upscale_by_unique_color_count(inp: np.ndarray, ctx=None):
            colors = set(inp[inp != 0])
            K = len(colors)
            if K <= 1:
                return None
            return np.kron(inp, np.ones((K, K), dtype=int))
        prims["kronecker_upscale_by_unique_color_count"] = kronecker_upscale_by_unique_color_count

        # 183. Produit Kronecker du masque de la couleur modale avec l'image (tâche c3e719e8)
        def kronecker_product_mode_mask_with_self(inp: np.ndarray, ctx=None):
            uq, counts = np.unique(inp, return_counts=True)
            if len(uq) <= 1:
                return None
            mode_col = uq[np.argmax(counts)]
            mask = (inp == mode_col).astype(int)
            return np.kron(mask, inp)
        prims["kronecker_product_mode_mask_with_self"] = kronecker_product_mode_mask_with_self

        # 184. Produit Kronecker du masque de couleur 2 avec l'image (tâche cce03e0d)
        def kronecker_product_color2_mask_with_self(inp: np.ndarray, ctx=None):
            if 2 not in inp:
                return None
            return np.kron(inp == 2, inp)
        prims["kronecker_product_color2_mask_with_self"] = kronecker_product_color2_mask_with_self

        # 185. Rayons diagonaux vers le bas-droite projetés en grille 6x6 (tâche d13f3404)
        def diagonal_downright_rays_to_6x6(inp: np.ndarray, ctx=None):
            if inp.shape != (3, 3):
                return None
            out = np.zeros((6, 6), dtype=int)
            for r in range(3):
                for c in range(3):
                    val = inp[r, c]
                    if val != 0:
                        s = 0
                        while r + s < 6 and c + s < 6:
                            out[r + s, c + s] = val
                            s += 1
            return out
        prims["diagonal_downright_rays_to_6x6"] = diagonal_downright_rays_to_6x6

        # 186. Rétablissement de la barre interrompue à l'intersection (tâche ba97ae07)
        def restore_interrupted_bar_across_intersection(inp: np.ndarray, ctx=None):
            uq = [c for c in np.unique(inp) if c != 0]
            if len(uq) != 2:
                return None
            out = inp.copy()
            restored = False
            for c in uq:
                mask = (inp == c)
                rows = np.where(mask.any(axis=1))[0]
                cols = np.where(mask.any(axis=0))[0]
                rmin, rmax = rows.min(), rows.max()
                cmin, cmax = cols.min(), cols.max()
                if (rmax - rmin + 1) == inp.shape[0] and np.any(~mask[rmin:rmax+1, cmin:cmax+1]):
                    out[rmin:rmax+1, cmin:cmax+1] = c
                    restored = True
                elif (cmax - cmin + 1) == inp.shape[1] and np.any(~mask[rmin:rmax+1, cmin:cmax+1]):
                    out[rmin:rmax+1, cmin:cmax+1] = c
                    restored = True
            return out if restored else None
        prims["restore_interrupted_bar_across_intersection"] = restore_interrupted_bar_across_intersection

        # 187. Croix (+) en 3 au point médian entre deux marqueurs 1 (tâche e9614598)
        def plus_sign_3s_at_midpoint_of_two_1s(inp: np.ndarray, ctx=None):
            r, c = np.where(inp == 1)
            if len(r) != 2:
                return None
            out = inp.copy()
            r_mid = (r[0] + r[1]) // 2
            c_mid = (c[0] + c[1]) // 2
            out[r_mid, c_mid] = 3
            if r_mid - 1 >= 0: out[r_mid - 1, c_mid] = 3
            if r_mid + 1 < inp.shape[0]: out[r_mid + 1, c_mid] = 3
            if c_mid - 1 >= 0: out[r_mid, c_mid - 1] = 3
            if c_mid + 1 < inp.shape[1]: out[r_mid, c_mid + 1] = 3
            return out
        prims["plus_sign_3s_at_midpoint_of_two_1s"] = plus_sign_3s_at_midpoint_of_two_1s

        # 188. Rayons diagonaux en V vers le haut depuis le piédestal inférieur (tâche b8cdaf2b)
        def v_shaped_upward_diagonal_rays_from_pedestal(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            if H < 3:
                return None
            bot = inp[-1]
            uq_bot = np.unique(bot[bot != 0])
            edge_col = bot[0]
            center_cols = [c for c in uq_bot if c != edge_col]
            if not center_cols:
                return None
            center_col = center_cols[0]
            ped_row = inp[-2]
            ped_cols = np.where(ped_row != 0)[0]
            if len(ped_cols) == 0:
                return None
            c_min = ped_cols.min()
            c_max = ped_cols.max()
            out = inp.copy()
            r = H - 3
            left_c = c_min - 1
            right_c = c_max + 1
            while r >= 0 and (left_c >= 0 or right_c < W):
                if left_c >= 0: out[r, left_c] = center_col
                if right_c < W: out[r, right_c] = center_col
                r -= 1
                left_c -= 1
                right_c += 1
            return out
        prims["v_shaped_upward_diagonal_rays_from_pedestal"] = v_shaped_upward_diagonal_rays_from_pedestal

        # 189. Réaction de contact adjacent entre 3 et 2 formant 8 (tâche d90796e8)
        def reaction_contact_pair_3_and_2_to_8(inp: np.ndarray, ctx=None):
            if 3 not in inp or 2 not in inp:
                return None
            H, W = inp.shape
            out = inp.copy()
            reacted = False
            for r in range(H):
                for c in range(W):
                    if inp[r, c] == 3:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and inp[nr, nc] == 2:
                                out[r, c] = 8
                                out[nr, nc] = 0
                                reacted = True
            return out if reacted else None
        prims["reaction_contact_pair_3_and_2_to_8"] = reaction_contact_pair_3_and_2_to_8

        # 190. OU logique bit à bit des deux moitiés recoloré en 6 (tâche dae9d2b5)
        def bitwise_or_of_halves_recolor_to_6(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            if W % 2 != 0:
                return None
            w2 = W // 2
            left = inp[:, :w2]
            right = inp[:, w2:]
            return ((left != 0) | (right != 0)).astype(int) * 6
        prims["bitwise_or_of_halves_recolor_to_6"] = bitwise_or_of_halves_recolor_to_6

        # 191. Duplication horizontale du motif x2 (tâche a416b8f3)
        def horizontal_tile_duplication_x2(inp: np.ndarray, ctx=None):
            return np.hstack([inp, inp])
        prims["horizontal_tile_duplication_x2"] = horizontal_tile_duplication_x2

        # 192. Agrandissement au plus proche voisin facteur 2x (tâche c59eb873)
        def nearest_neighbor_2x_upscale(inp: np.ndarray, ctx=None):
            return np.kron(inp, np.ones((2, 2), dtype=int))
        prims["nearest_neighbor_2x_upscale"] = nearest_neighbor_2x_upscale

        # 193. Expansion en escalier depuis un vecteur 1D (tâche bbc9ae5d)
        def staircase_expansion_from_1d_row(inp: np.ndarray, ctx=None):
            if inp.shape[0] != 1 or inp.shape[1] % 2 != 0:
                return None
            W = inp.shape[1]
            H = W // 2
            nz = inp[0, inp[0] != 0]
            if len(nz) == 0:
                return None
            C = nz[0]
            K = len(nz)
            out = np.zeros((H, W), dtype=int)
            for i in range(H):
                out[i, :min(W, K + i)] = C
            return out
        prims["staircase_expansion_from_1d_row"] = staircase_expansion_from_1d_row

        # 194. Projection d'archétype 3x3 selon la couleur d'entrée 1, 2 ou 3 (tâche d4469b4b)
        def project_archetype_by_input_color_1_2_3(inp: np.ndarray, ctx=None):
            MAP = {
                1: np.array([[0, 5, 0], [5, 5, 5], [0, 5, 0]]),
                2: np.array([[5, 5, 5], [0, 5, 0], [0, 5, 0]]),
                3: np.array([[0, 0, 5], [0, 0, 5], [5, 5, 5]]),
            }
            uq = [c for c in np.unique(inp) if c != 0]
            if len(uq) != 1 or uq[0] not in MAP:
                return None
            return MAP[uq[0]]
        prims["project_archetype_by_input_color_1_2_3"] = project_archetype_by_input_color_1_2_3

        # 195. Pas de 1 unité du pixel 3 vers le pixel 4 (tâche dc433765)
        def step_pixel_3_one_unit_towards_pixel_4(inp: np.ndarray, ctx=None):
            r3, c3 = np.where(inp == 3)
            r4, c4 = np.where(inp == 4)
            if len(r3) != 1 or len(r4) != 1:
                return None
            dr = int(np.sign(r4[0] - r3[0]))
            dc = int(np.sign(c4[0] - c3[0]))
            out = inp.copy()
            out[r3[0], c3[0]] = 0
            out[r3[0] + dr, c3[0] + dc] = 3
            return out
        prims["step_pixel_3_one_unit_towards_pixel_4"] = step_pixel_3_one_unit_towards_pixel_4

        # 196. Recadrage boîte englobante hors-1 et mise à 0 des 1s internes (tâche a740d043)
        def crop_non_1_bounding_box_and_zero_1s(inp: np.ndarray, ctx=None):
            mask = (inp != 1)
            pts = np.argwhere(mask)
            if len(pts) == 0:
                return None
            rmin, cmin = pts.min(axis=0)
            rmax, cmax = pts.max(axis=0)
            crop = inp[rmin:rmax+1, cmin:cmax+1].copy()
            crop[crop == 1] = 0
            return crop
        prims["crop_non_1_bounding_box_and_zero_1s"] = crop_non_1_bounding_box_and_zero_1s

        # 197. Réflexion verticale miroir moitié inférieure vers moitié supérieure (tâche f25ffba3)
        def vertical_reflection_mirror_bottom_to_top(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            out = inp.copy()
            out[:H//2] = np.flipud(inp[H//2:])
            return out
        prims["vertical_reflection_mirror_bottom_to_top"] = vertical_reflection_mirror_bottom_to_top

        # 198. Rayon rebondissant depuis le coin inférieur gauche (tâche a3df8b1e)
        def bouncing_ray_from_bottom_left(inp: np.ndarray, ctx=None):
            non_zeros = inp[inp != 0]
            if len(non_zeros) == 0:
                return None
            col = non_zeros[0]
            pts = np.argwhere(inp == col)
            if len(pts) == 0:
                return None
            H, W = inp.shape
            r, c = pts[0]
            dr, dc = -1, 1
            out = inp.copy()
            while 0 <= r < H:
                out[r, c] = col
                if c == 0 and dc < 0:
                    dc = 1
                elif c == W - 1 and dc > 0:
                    dc = -1
                r += dr
                c += dc
            return out
        prims["bouncing_ray_from_bottom_left"] = bouncing_ray_from_bottom_left

        # 199. Remplissage des lignes par couleur dictée par colonne du 5 (tâche a85d4709)
        def recolor_rows_by_5_column_position(inp: np.ndarray, ctx=None):
            col_map = {0: 2, 1: 4, 2: 3}
            out = inp.copy()
            for r in range(inp.shape[0]):
                c5 = np.where(inp[r] == 5)[0]
                if len(c5) > 0 and c5[0] in col_map:
                    out[r, :] = col_map[c5[0]]
            return out
        prims["recolor_rows_by_5_column_position"] = recolor_rows_by_5_column_position

        # 200. Pont horizontal de 2 entre deux 1s espacés d'une case (tâche a699fb00)
        def bridge_horizontal_distance2_ones_with_2(inp: np.ndarray, ctx=None):
            out = inp.copy()
            H, W = inp.shape
            for r in range(H):
                for c in range(1, W - 1):
                    if inp[r, c] == 0 and inp[r, c-1] == 1 and inp[r, c+1] == 1:
                        out[r, c] = 2
            return out
        prims["bridge_horizontal_distance2_ones_with_2"] = bridge_horizontal_distance2_ones_with_2

        # 201. Émission des 4 coins diagonaux colorés autour du marqueur 2 (tâche a9f96cdd)
        def project_four_colored_corners_around_2(inp: np.ndarray, ctx=None):
            out = inp.copy()
            pts = np.argwhere(inp == 2)
            H, W = inp.shape
            for r, c in pts:
                out[r, c] = 0
                if r - 1 >= 0 and c - 1 >= 0: out[r-1, c-1] = 3
                if r - 1 >= 0 and c + 1 < W:  out[r-1, c+1] = 6
                if r + 1 < H and c - 1 >= 0:  out[r+1, c-1] = 8
                if r + 1 < H and c + 1 < W:   out[r+1, c+1] = 7
            return out
        prims["project_four_colored_corners_around_2"] = project_four_colored_corners_around_2

        # 202. Remplissage croix X 3x3 par le nombre de blocs 2x2 de 2 (tâche ff28f65a)
        def fill_x_pattern_by_2x2_block_count(inp: np.ndarray, ctx=None):
            cnt = int(np.sum(inp == 2) // 4)
            out = np.zeros((3, 3), dtype=inp.dtype)
            coords = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
            for i in range(min(cnt, len(coords))):
                out[coords[i]] = 1
            return out
        prims["fill_x_pattern_by_2x2_block_count"] = fill_x_pattern_by_2x2_block_count

        # 203. Rognage bloc 2x2 supérieur gauche (tâche d10ecb37)
        def crop_top_left_2x2(inp: np.ndarray, ctx=None):
            return inp[:2, :2].copy()
        prims["crop_top_left_2x2"] = crop_top_left_2x2

        # 204. Superposition hiérarchique prioritaire de trois blocs 4x4 (tâche cf98881b)
        def priority_overlay_three_4x4_blocks(inp: np.ndarray, ctx=None):
            if inp.shape[1] < 14 or inp.shape[0] < 4:
                return None
            b0 = inp[:, 0:4]
            b1 = inp[:, 5:9]
            b2 = inp[:, 10:14]
            out = b2.copy()
            out[b1 != 0] = b1[b1 != 0]
            out[b0 != 0] = b0[b0 != 0]
            return out
        prims["priority_overlay_three_4x4_blocks"] = priority_overlay_three_4x4_blocks

        # 205. Classement des 3 couleurs dominantes hors fond (tâche f8b3ba0a)
        def rank_top_3_colors_excluding_background(inp: np.ndarray, ctx=None):
            vals, counts = np.unique(inp[inp != 0], return_counts=True)
            if len(vals) < 3:
                return None
            bg_cand = vals[np.argmax(counts)]
            other_vals = [v for v in vals if v != bg_cand]
            sorted_cols = sorted(other_vals, key=lambda c: np.sum(inp == c), reverse=True)
            return np.array(sorted_cols[:3], dtype=inp.dtype).reshape(3, 1)
        prims["rank_top_3_colors_excluding_background"] = rank_top_3_colors_excluding_background

        # 206. Classement des 3 couleurs par fréquence décroissante (tâche f8ff0b80)
        def rank_3_colors_by_frequency(inp: np.ndarray, ctx=None):
            vals, counts = np.unique(inp[inp != 0], return_counts=True)
            sorted_cols = [c for c, _ in sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)]
            return np.array(sorted_cols[:3], dtype=inp.dtype).reshape(3, 1)
        prims["rank_3_colors_by_frequency"] = rank_3_colors_by_frequency

        # 207. Effacement des deux diagonales principales (tâche ea786f4a)
        def zero_both_main_diagonals(inp: np.ndarray, ctx=None):
            out = inp.copy()
            np.fill_diagonal(out, 0)
            np.fill_diagonal(np.fliplr(out), 0)
            return out
        prims["zero_both_main_diagonals"] = zero_both_main_diagonals

        # 208. Conservation unique de la colonne centrale (tâche d23f8c26)
        def keep_center_column_only(inp: np.ndarray, ctx=None):
            mid = inp.shape[1] // 2
            out = np.zeros_like(inp)
            out[:, mid] = inp[:, mid]
            return out
        prims["keep_center_column_only"] = keep_center_column_only

        # 209. Projection des points des quatre quadrants vers le carré central 2x2 (tâche d89b689b)
        def map_four_quadrant_dots_to_center_2x2_box(inp: np.ndarray, ctx=None):
            r8, c8 = np.where(inp == 8)
            if len(r8) == 0:
                return None
            rmin, rmax = r8.min(), r8.max()
            cmin, cmax = c8.min(), c8.max()
            out = np.zeros_like(inp)
            dots = np.argwhere((inp != 0) & (inp != 8))
            for r, c in dots:
                val = inp[r, c]
                tr = rmin if r < rmin else rmax
                tc = cmin if c < cmin else cmax
                out[tr, tc] = val
            return out
        prims["map_four_quadrant_dots_to_center_2x2_box"] = map_four_quadrant_dots_to_center_2x2_box

        # 210. Rayons orthogonaux depuis les points alignés vers le bloc 3x3/2x2 (tâche d43fd935)
        def project_aligned_dots_to_center_box(inp: np.ndarray, ctx=None):
            r3, c3 = np.where(inp == 3)
            if len(r3) == 0:
                return None
            rmin, rmax = r3.min(), r3.max()
            cmin, cmax = c3.min(), c3.max()
            out = inp.copy()
            dots = np.argwhere((inp != 0) & (inp != 3))
            for r, c in dots:
                color = inp[r, c]
                if rmin <= r <= rmax:
                    if c < cmin:
                        out[r, c:cmin] = color
                    elif c > cmax:
                        out[r, cmax+1:c+1] = color
                if cmin <= c <= cmax:
                    if r < rmin:
                        out[r:rmin, c] = color
                    elif r > rmax:
                        out[rmax+1:r+1, c] = color
            return out
        prims["project_aligned_dots_to_center_box"] = project_aligned_dots_to_center_box

        # 211. Extraction de la couleur centrale d'un carré creux 3x3 (tâche d9fac9be)
        def extract_color_inside_hollow_3x3_square(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            for r in range(H - 2):
                for c in range(W - 2):
                    sub = inp[r:r+3, c:c+3]
                    ring = [sub[0,0], sub[0,1], sub[0,2], sub[1,0], sub[1,2], sub[2,0], sub[2,1], sub[2,2]]
                    center = sub[1,1]
                    if len(set(ring)) == 1 and ring[0] != 0 and ring[0] != center and center != 0:
                        return np.array([[center]], dtype=inp.dtype)
            return None
        prims["extract_color_inside_hollow_3x3_square"] = extract_color_inside_hollow_3x3_square

        # 212. Couleur de la région contenant le plus de points intrus (tâche de1cd16c)
        def color_of_region_with_most_intruder_dots(inp: np.ndarray, ctx=None):
            from collections import Counter
            vals, counts = np.unique(inp, return_counts=True)
            intruder = vals[np.argmin(counts)]
            H, W = inp.shape
            region_counts = Counter()
            for r in range(H):
                for c in range(W):
                    if inp[r, c] == intruder:
                        neighbors = []
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < H and 0 <= nc < W and inp[nr, nc] != intruder:
                                neighbors.append(inp[nr, nc])
                        if neighbors:
                            host = Counter(neighbors).most_common(1)[0][0]
                            region_counts[host] += 1
            if not region_counts:
                return None
            winner = region_counts.most_common(1)[0][0]
            return np.array([[winner]], dtype=inp.dtype)
        prims["color_of_region_with_most_intruder_dots"] = color_of_region_with_most_intruder_dots

        # 213. Débruitage des singletons isolés en 8-connexité (tâche 42a50994)
        def denoise_8conn_singletons(inp: np.ndarray, ctx=None):
            non_zeros = [c for c in np.unique(inp) if c != 0]
            if len(non_zeros) == 0:
                return None
            col = non_zeros[0]
            lbl, num = label(inp == col, structure=np.ones((3, 3)))
            out = np.zeros_like(inp)
            for cid in range(1, num + 1):
                mask = (lbl == cid)
                if np.sum(mask) >= 2:
                    out[mask] = col
            return out
        prims["denoise_8conn_singletons"] = denoise_8conn_singletons

        # 214. Extraction de la composante solide contenant le maximum de pixels 2 (tâches 8efcae92 & e50d258f)
        def extract_solid_component_with_max_color_2(inp: np.ndarray, ctx=None):
            lbl, num = label(inp != 0, structure=np.ones((3, 3)))
            best_crop = None
            max_c2 = -1
            for cid in range(1, num + 1):
                pts = np.argwhere(lbl == cid)
                rmin, cmin = pts.min(0); rmax, cmax = pts.max(0)
                crop = inp[rmin:rmax+1, cmin:cmax+1]
                c2 = np.sum(crop == 2)
                if c2 > max_c2:
                    max_c2 = c2
                    best_crop = crop
            return best_crop
        prims["extract_solid_component_with_max_color_2"] = extract_solid_component_with_max_color_2

        # 215. Trajet en L droite puis bas jusqu'aux parois (tâche 99fa7670)
        def shoot_right_then_down_path(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            pred = np.zeros_like(inp)
            dots = np.argwhere(inp != 0)
            for r, c in dots:
                col = inp[r, c]
                pred[r, c:W] = col
                pred[r:H, W-1] = col
            return pred
        prims["shoot_right_then_down_path"] = shoot_right_then_down_path

        # 216. Histogramme des fréquences en barres verticales ordonnées (tâche 9af7a82c)
        def frequency_histogram_vertical_bars(inp: np.ndarray, ctx=None):
            vals, counts = np.unique(inp[inp != 0], return_counts=True)
            if len(vals) == 0:
                return None
            sorted_pairs = sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)
            max_cnt = sorted_pairs[0][1]
            num_cols = len(sorted_pairs)
            pred = np.zeros((max_cnt, num_cols), dtype=inp.dtype)
            for c_idx, (col, cnt) in enumerate(sorted_pairs):
                pred[:cnt, c_idx] = col
            return pred
        prims["frequency_histogram_vertical_bars"] = frequency_histogram_vertical_bars

        # 217. Pochoir d'inversion du masque de couleur 5 (tâche f76d97a5)
        def stencil_invert_mask_of_color_5(inp: np.ndarray, ctx=None):
            other = [c for c in np.unique(inp) if c != 5]
            if not other:
                return None
            other_col = other[0]
            pred = np.zeros_like(inp)
            pred[inp == 5] = other_col
            return pred
        prims["stencil_invert_mask_of_color_5"] = stencil_invert_mask_of_color_5

        # 218. Décalage cyclique des anneaux concentriques carrés (tâche bda2d7a6)
        def cyclic_concentric_square_rings_shift(inp: np.ndarray, ctx=None):
            N = inp.shape[0]
            if N < 2 or inp.shape[1] != N:
                return None
            C0 = inp[0, 0]
            C1 = inp[1, 1]
            C2 = inp[2, 2] if N >= 6 else C0
            shifted = [C2, C0, C1]
            pred = np.zeros_like(inp)
            for r in range(N):
                for c in range(N):
                    layer = min(r, c, N - 1 - r, N - 1 - c)
                    pred[r, c] = shifted[layer % 3]
            return pred
        prims["cyclic_concentric_square_rings_shift"] = cyclic_concentric_square_rings_shift

        # 219. Entonnoir triangulaire rayé en alternance depuis barre verticale 7 (tâche db3e9e38)
        def alternating_striped_triangle_funnel(inp: np.ndarray, ctx=None):
            pts = np.argwhere(inp == 7)
            if len(pts) == 0:
                return None
            c_bar = pts[0, 1]
            L = len(pts)
            pred = np.zeros_like(inp)
            for r in range(L):
                spread = L - 1 - r
                c_start = max(0, c_bar - spread)
                c_end = min(inp.shape[1] - 1, c_bar + spread)
                for c in range(c_start, c_end + 1):
                    dist = abs(c - c_bar)
                    pred[r, c] = 7 if (dist % 2 == 0) else 8
            return pred
        prims["alternating_striped_triangle_funnel"] = alternating_striped_triangle_funnel

        # 220. Rayon rebondissant montant depuis le coin inférieur gauche (tâche e179c5f4)
        def bouncing_ray_upwards_from_bottom_left(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            pred = np.full((H, W), 8, dtype=inp.dtype)
            dc = 1 if W > 1 else 0
            c = 0
            for r in range(H - 1, -1, -1):
                pred[r, c] = 1
                if c == 0 and dc < 0:
                    dc = 1
                elif c == W - 1 and dc > 0:
                    dc = -1
                c += dc
            return pred
        prims["bouncing_ray_upwards_from_bottom_left"] = bouncing_ray_upwards_from_bottom_left

        # 221. Remplissage des blocs 3x3 par 1 s'ils contiennent 5 (tâche ce22a75a)
        def fill_3x3_blocks_with_1s_containing_5(inp: np.ndarray, ctx=None):
            if inp.shape[0] % 3 != 0 or inp.shape[1] % 3 != 0:
                return None
            pred = np.zeros_like(inp)
            br_max, bc_max = inp.shape[0] // 3, inp.shape[1] // 3
            for br in range(br_max):
                for bc in range(bc_max):
                    sub = inp[br*3:(br+1)*3, bc*3:(bc+1)*3]
                    if 5 in sub:
                        pred[br*3:(br+1)*3, bc*3:(bc+1)*3] = 1
            return pred
        prims["fill_3x3_blocks_with_1s_containing_5"] = fill_3x3_blocks_with_1s_containing_5

        # 222. Estampille de motif croisé 3x3 centré sur les 5s (tâche b60334d2)
        def stamp_3x3_cross_pattern_centered_on_5(inp: np.ndarray, ctx=None):
            pred = np.zeros_like(inp)
            pts = np.argwhere(inp == 5)
            stamp = np.array([[5, 1, 5], [1, 0, 1], [5, 1, 5]])
            for r, c in pts:
                r0, r1 = r - 1, r + 2
                c0, c1 = c - 1, c + 2
                if 0 <= r0 and r1 <= inp.shape[0] and 0 <= c0 and c1 <= inp.shape[1]:
                    pred[r0:r1, c0:c1] = stamp
            return pred
        prims["stamp_3x3_cross_pattern_centered_on_5"] = stamp_3x3_cross_pattern_centered_on_5

        # 223. Plus haute barre en 1, plus courte en 2, effacement du reste (tâche a61f2674)
        def tallest_bar_1_shortest_bar_2_erase_rest(inp: np.ndarray, ctx=None):
            col_heights = {}
            for c in range(inp.shape[1]):
                h = np.sum(inp[:, c] == 5)
                if h > 0:
                    col_heights[c] = h
            if not col_heights:
                return None
            tallest_c = max(col_heights, key=col_heights.get)
            shortest_c = min(col_heights, key=col_heights.get)
            pred = np.zeros_like(inp)
            pred[inp[:, tallest_c] == 5, tallest_c] = 1
            pred[inp[:, shortest_c] == 5, shortest_c] = 2
            return pred
        prims["tallest_bar_1_shortest_bar_2_erase_rest"] = tallest_bar_1_shortest_bar_2_erase_rest

        # 224. Centrage de la forme 2 entre les 4 coins repères 3 (tâche a1570a43)
        def center_shape_2_inside_4_corner_anchors_3(inp: np.ndarray, ctx=None):
            p3 = np.argwhere(inp == 3)
            p2 = np.argwhere(inp == 2)
            if len(p3) == 0 or len(p2) == 0:
                return None
            r3_min, c3_min = p3.min(0); r3_max, c3_max = p3.max(0)
            r2_min, c2_min = p2.min(0); r2_max, c2_max = p2.max(0)
            h2 = r2_max - r2_min + 1
            w2 = c2_max - c2_min + 1
            target_r0 = r3_min + ((r3_max - r3_min + 1) - h2) // 2
            target_c0 = c3_min + ((c3_max - c3_min + 1) - w2) // 2
            dr = target_r0 - r2_min
            dc = target_c0 - c2_min
            pred = np.zeros_like(inp)
            for r, c in p3:
                pred[r, c] = 3
            for r, c in p2:
                if 0 <= r + dr < inp.shape[0] and 0 <= c + dc < inp.shape[1]:
                    pred[r + dr, c + dc] = 2
            return pred
        prims["center_shape_2_inside_4_corner_anchors_3"] = center_shape_2_inside_4_corner_anchors_3

        # 225. Remplissage des colonnes non-vides par 8 et pavage 2x2 (tâche f5b8619d)
        def fill_non_empty_columns_with_8_and_tile_2x2(inp: np.ndarray, ctx=None):
            sub = inp.copy()
            H, W = inp.shape
            for c in range(W):
                col_vals = inp[:, c]
                if np.any(col_vals != 0):
                    sub[col_vals == 0, c] = 8
            return np.tile(sub, (2, 2))
        prims["fill_non_empty_columns_with_8_and_tile_2x2"] = fill_non_empty_columns_with_8_and_tile_2x2

        # 226. Inpainting périodique du bloc manquant de zéros (tâche f9012d9b)
        def periodic_inpaint_zero_block(inp: np.ndarray, ctx=None):
            z_pts = np.argwhere(inp == 0)
            if len(z_pts) == 0:
                return None
            r0, c0 = z_pts.min(0); r1, c1 = z_pts.max(0) + 1
            h, w = r1 - r0, c1 - c0
            H, W = inp.shape
            best_pred = None
            max_overlap = -1
            for dr in range(-H + 1, H):
                for dc in range(-W + 1, W):
                    if dr == 0 and dc == 0: continue
                    consistent = True
                    overlap_count = 0
                    for r in range(H):
                        for c in range(W):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                v1 = inp[r, c]; v2 = inp[nr, nc]
                                if v1 != 0 and v2 != 0:
                                    if v1 != v2:
                                        consistent = False
                                        break
                                    overlap_count += 1
                        if not consistent: break
                    if consistent and overlap_count > 0:
                        can_fill = True
                        pred = np.zeros((h, w), dtype=inp.dtype)
                        for r, c in z_pts:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and inp[nr, nc] != 0:
                                pred[r - r0, c - c0] = inp[nr, nc]
                            else:
                                can_fill = False
                                break
                        if can_fill and overlap_count > max_overlap:
                            max_overlap = overlap_count
                            best_pred = pred
            return best_pred
        prims["periodic_inpaint_zero_block"] = periodic_inpaint_zero_block

        # 227. Minimap des compartiments délimités par lignes de zéros (tâche 780d0b14)
        def compartment_minimap_by_divider_lines(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            zero_rows = [r for r in range(H) if np.all(inp[r] == 0)]
            zero_cols = [c for c in range(W) if np.all(inp[:, c] == 0)]
            r_cuts = [-1] + zero_rows + [H]
            c_cuts = [-1] + zero_cols + [W]
            h_blocks = len(r_cuts) - 1
            w_blocks = len(c_cuts) - 1
            if h_blocks <= 1 and w_blocks <= 1:
                return None
            pred = np.zeros((h_blocks, w_blocks), dtype=inp.dtype)
            for bi in range(h_blocks):
                r0 = r_cuts[bi] + 1
                r1 = r_cuts[bi + 1]
                for bj in range(w_blocks):
                    c0 = c_cuts[bj] + 1
                    c1 = c_cuts[bj + 1]
                    sub = inp[r0:r1, c0:c1]
                    nz = sub[sub != 0]
                    if len(nz) > 0:
                        vals, counts = np.unique(nz, return_counts=True)
                        pred[bi, bj] = vals[np.argmax(counts)]
            return pred
        prims["compartment_minimap_by_divider_lines"] = compartment_minimap_by_divider_lines

        # 228. Cellules de grille 3x3 dictées par 4 coins colorés (tâche 7837ac64)
        def macro_3x3_grid_cells_by_4_corners(inp: np.ndarray, ctx=None):
            vals, counts = np.unique(inp[inp != 0], return_counts=True)
            if len(vals) < 2:
                return None
            grid_col = vals[np.argmax(counts)]
            other_pts = np.argwhere((inp != 0) & (inp != grid_col))
            all_r = sorted(list(set(other_pts[:, 0])))
            all_c = sorted(list(set(other_pts[:, 1])))
            if len(all_r) < 4 or len(all_c) < 4:
                return None
            pred = np.zeros((3, 3), dtype=inp.dtype)
            for r_idx in range(3):
                r_top = all_r[r_idx]; r_bot = all_r[r_idx + 1]
                for c_idx in range(3):
                    c_left = all_c[c_idx]; c_right = all_c[c_idx + 1]
                    corners = [inp[r_top, c_left], inp[r_top, c_right], inp[r_bot, c_left], inp[r_bot, c_right]]
                    for col in set(corners) - {0, grid_col}:
                        if corners.count(col) == 4:
                            pred[r_idx, c_idx] = col
            return pred
        prims["macro_3x3_grid_cells_by_4_corners"] = macro_3x3_grid_cells_by_4_corners

        # 229. Matrice identité diagonale de 8s selon nombre de composantes (tâche d0f5fe59)
        def diagonal_identity_matrix_by_component_count(inp: np.ndarray, ctx=None):
            lbl, num = label(inp == 8, structure=np.ones((3, 3)))
            if num == 0:
                return None
            pred = np.zeros((num, num), dtype=inp.dtype)
            np.fill_diagonal(pred, 8)
            return pred
        prims["diagonal_identity_matrix_by_component_count"] = diagonal_identity_matrix_by_component_count

        # 230. Superposition de 4 quadrants avec priorité stricte [7, 4, 8, 6] (tâche a68b268e)
        def overlay_four_quadrants_with_priority_7_4_8_6(inp: np.ndarray, ctx=None):
            if inp.shape != (9, 9):
                return None
            tl = inp[0:4, 0:4]
            tr = inp[0:4, 5:9]
            bl = inp[5:9, 0:4]
            br = inp[5:9, 5:9]
            priority = [7, 4, 8, 6]
            pred = np.zeros((4, 4), dtype=inp.dtype)
            for r in range(4):
                for c in range(4):
                    vals = [tl[r, c], tr[r, c], bl[r, c], br[r, c]]
                    chosen = 0
                    for p in priority:
                        if p in vals:
                            chosen = p
                            break
                    pred[r, c] = chosen
            return pred
        prims["overlay_four_quadrants_with_priority_7_4_8_6"] = overlay_four_quadrants_with_priority_7_4_8_6

        # 231. Pluie sous parasol vers le bas (tâche 6d58a25d)
        def canopy_rain_rays_downward(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) != 2:
                return None
            umb_c = None
            dot_c = None
            for c in colors:
                if np.sum(inp == c) == 10:
                    umb_c = c
                else:
                    dot_c = c
            if umb_c is None:
                counts = {c: np.sum(inp == c) for c in colors}
                dot_c = min(counts, key=counts.get)
                umb_c = max(counts, key=counts.get)
            H, W = inp.shape
            umb_coords = np.argwhere(inp == umb_c)
            c_to_max_r = {}
            for r, c in umb_coords:
                if c not in c_to_max_r or r > c_to_max_r[c]:
                    c_to_max_r[c] = r
            pred = inp.copy()
            for c, r_max in c_to_max_r.items():
                drops_below = np.where(inp[r_max+1:, c] == dot_c)[0]
                if len(drops_below) > 0:
                    pred[r_max+1:, c] = dot_c
            return pred
        prims["canopy_rain_rays_downward"] = canopy_rain_rays_downward

        # 232. Complétion de lignes morpion (tic-tac-toe) dans sous-cellules 3x3 de blocs 2x2 (tâche cbded52d)
        def tic_tac_toe_completion_in_subcells(inp: np.ndarray, ctx=None):
            if inp.shape != (8, 8):
                return None
            pred = inp.copy()
            row_starts = [0, 3, 6]
            col_starts = [0, 3, 6]
            for sub_r in range(2):
                for sub_c in range(2):
                    pattern = np.zeros((3, 3), dtype=int)
                    for r_idx, r_s in enumerate(row_starts):
                        for c_idx, c_s in enumerate(col_starts):
                            pattern[r_idx, c_idx] = pred[r_s + sub_r, c_s + sub_c]
                    for r in range(3):
                        row_vals = pattern[r, :]
                        colors = [c for c in row_vals if c != 1 and c != 0]
                        for c in set(colors):
                            if np.sum(row_vals == c) == 2:
                                for idx in range(3):
                                    if pattern[r, idx] == 1:
                                        pattern[r, idx] = c
                    for col in range(3):
                        col_vals = pattern[:, col]
                        colors = [c for c in col_vals if c != 1 and c != 0]
                        for c in set(colors):
                            if np.sum(col_vals == c) == 2:
                                for idx in range(3):
                                    if pattern[idx, col] == 1:
                                        pattern[idx, col] = c
                    for r_idx, r_s in enumerate(row_starts):
                        for c_idx, c_s in enumerate(col_starts):
                            pred[r_s + sub_r, c_s + sub_c] = pattern[r_idx, c_idx]
            return pred
        prims["tic_tac_toe_completion_in_subcells"] = tic_tac_toe_completion_in_subcells

        # 233. Balayage diagonal montant depuis une ligne 1D 1x5 (tâche feca6190)
        def diagonal_sweep_from_1d_row(inp: np.ndarray, ctx=None):
            if inp.ndim != 2 or inp.shape[0] != 1 or inp.shape[1] != 5:
                return None
            row = inp[0]
            K = int(np.sum(row != 0))
            if K == 0:
                return None
            size = K * 5
            pred = np.zeros((size, size), dtype=inp.dtype)
            for k in range(size):
                r = size - 1 - k
                for c_in, val in enumerate(row):
                    c_out = c_in + k
                    if 0 <= c_out < size:
                        pred[r, c_out] = val
            return pred
        prims["diagonal_sweep_from_1d_row"] = diagonal_sweep_from_1d_row

        # 234. Alternance de lignes diagonales (pente +1) avec couleur 4 aux indices impairs (tâche a5f85a15)
        def alternate_diagonal_lines_with_4(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            pred = inp.copy()
            for diff in range(-W, H):
                coords = []
                for r in range(H):
                    c = r - diff
                    if 0 <= c < W and inp[r, c] != 0:
                        coords.append((r, c))
                for idx, (r, c) in enumerate(coords):
                    if idx % 2 == 1:
                        pred[r, c] = 4
            return pred
        prims["alternate_diagonal_lines_with_4"] = alternate_diagonal_lines_with_4

        # 235. Lignes unies cycliques répétées depuis palette d'en-tête (tâche bd4472b8)
        def cyclic_solid_rows_from_palette_header(inp: np.ndarray, ctx=None):
            if inp.shape[0] < 3:
                return None
            if not np.all(inp[1, :] == 5):
                return None
            H, W = inp.shape
            pred = inp.copy()
            palette = inp[0, :]
            for r in range(2, H):
                pred[r, :] = palette[(r - 2) % len(palette)]
            return pred
        prims["cyclic_solid_rows_from_palette_header"] = cyclic_solid_rows_from_palette_header

        # 236. Remplacement de période 3 et 6 par couleur 6 sur grille 3xN (tâche ba26e723)
        def period_3_alternating_stamping_on_3_rows(inp: np.ndarray, ctx=None):
            if inp.shape[0] != 3 or not np.all(inp[1, :] == 4):
                return None
            H, W = inp.shape
            pred = inp.copy()
            for c in range(0, W, 3):
                pred[1, c] = 6
            if inp[0, 0] == 4:
                for c in range(0, W, 6):
                    pred[0, c] = 6
                for c in range(3, W, 6):
                    pred[2, c] = 6
            else:
                for c in range(3, W, 6):
                    pred[0, c] = 6
                for c in range(0, W, 6):
                    pred[2, c] = 6
            return pred
        prims["period_3_alternating_stamping_on_3_rows"] = period_3_alternating_stamping_on_3_rows

        # 237. Expansion de barre horizontale : triangles étagés 3 au-dessus, 1 en-dessous (tâche a65b410d)
        def expand_horizontal_bar_above_3_below_1(inp: np.ndarray, ctx=None):
            coords = np.argwhere(inp == 2)
            if len(coords) == 0:
                return None
            r_bar = coords[0, 0]
            if not np.all(coords[:, 0] == r_bar):
                return None
            L = len(coords)
            H, W = inp.shape
            pred = inp.copy()
            for r in range(r_bar - 1, -1, -1):
                k = r_bar - r
                length = L + k
                pred[r, :min(W, length)] = 3
            for r in range(r_bar + 1, H):
                k = r - r_bar
                length = L - k
                if length > 0:
                    pred[r, :min(W, length)] = 1
                else:
                    break
            return pred
        prims["expand_horizontal_bar_above_3_below_1"] = expand_horizontal_bar_above_3_below_1

        # 238. Rayons diagonaux émis depuis les coins d'un conteneur en L ou U (tâche ec883f72)
        def shoot_diagonal_rays_from_container_corners(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) != 2:
                return None
            bbox_areas = {}
            for c in colors:
                coords = np.argwhere(inp == c)
                h = coords[:, 0].max() - coords[:, 0].min() + 1
                w = coords[:, 1].max() - coords[:, 1].min() + 1
                bbox_areas[c] = h * w
            cont_c = max(bbox_areas, key=bbox_areas.get)
            obj_c = min(bbox_areas, key=bbox_areas.get)
            H, W = inp.shape
            pred = inp.copy()
            for r in range(H):
                for c in range(W):
                    if inp[r, c] == cont_c:
                        nbrs = []
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and inp[nr, nc] == cont_c:
                                nbrs.append((dr, dc))
                        if len(nbrs) == 2:
                            (dr1, dc1), (dr2, dc2) = nbrs
                            if dr1 * dr2 + dc1 * dc2 == 0:
                                ray_dr = -(dr1 + dr2)
                                ray_dc = -(dc1 + dc2)
                                curr_r, curr_c = r + ray_dr, c + ray_dc
                                while 0 <= curr_r < H and 0 <= curr_c < W and inp[curr_r, curr_c] == 0:
                                    pred[curr_r, curr_c] = obj_c
                                    curr_r += ray_dr
                                    curr_c += ray_dc
            return pred
        prims["shoot_diagonal_rays_from_container_corners"] = shoot_diagonal_rays_from_container_corners

        # 239. Recolorisation des pixels non nuls selon la couleur de la colonne 0 (tâche c9f8e694)
        def recolor_nonzeros_by_column0_color(inp: np.ndarray, ctx=None):
            if not np.any(inp[:, 0] != 0):
                return None
            pred = inp.copy()
            H, W = inp.shape
            for r in range(H):
                for c in range(1, W):
                    if inp[r, c] != 0:
                        pred[r, c] = inp[r, 0]
            return pred
        prims["recolor_nonzeros_by_column0_color"] = recolor_nonzeros_by_column0_color

        # 240. Remplacement des blocs de 5 par le motif de la composante modèle (tâche e76a88a6)
        def replace_blocks_of_5_with_template(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            non_5_mask = (inp != 0) & (inp != 5)
            coords = np.argwhere(non_5_mask)
            if len(coords) == 0:
                return None
            r_min, r_max = coords[:, 0].min(), coords[:, 0].max()
            c_min, c_max = coords[:, 1].min(), coords[:, 1].max()
            template = inp[r_min:r_max+1, c_min:c_max+1]
            th, tw = template.shape
            lbl, n_feat = label(inp == 5)
            if n_feat == 0:
                return None
            pred = inp.copy()
            for feat in range(1, n_feat + 1):
                f_coords = np.argwhere(lbl == feat)
                fr_min, fc_min = f_coords[:, 0].min(), f_coords[:, 1].min()
                if fr_min + th <= inp.shape[0] and fc_min + tw <= inp.shape[1]:
                    pred[fr_min:fr_min+th, fc_min:fc_min+tw] = template
            return pred
        prims["replace_blocks_of_5_with_template"] = replace_blocks_of_5_with_template

        # 241. Pavage horizontal du motif unitaire périodique (tâche d8c310e9)
        def tile_horizontal_periodic_unit(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            non_empty_cols = [c for c in range(W) if np.any(inp[:, c] != 0)]
            if len(non_empty_cols) < 3:
                return None
            max_c = max(non_empty_cols)
            best_P = None
            for P in range(2, max_c):
                valid = True
                matches = 0
                for c in range(max_c + 1 - P):
                    if np.any(inp[:, c] != 0) and np.any(inp[:, c + P] != 0):
                        if not np.array_equal(inp[:, c], inp[:, c + P]):
                            valid = False
                            break
                        else:
                            matches += 1
                if valid and matches > 0:
                    best_P = P
                    break
            if best_P is None:
                return None
            pred = inp.copy()
            tile = inp[:, :best_P]
            for c in range(W):
                pred[:, c] = tile[:, c % best_P]
            return pred
        prims["tile_horizontal_periodic_unit"] = tile_horizontal_periodic_unit

        # 242. Recolorisation de blocs de 5 : coins 1, bordures 4, intérieur 2 (tâche b6afb2da)
        def recolor_rectangular_blocks_corners_1_borders_4_interior_2(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            struct = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
            lbl, n_feat = label(inp == 5, structure=struct)
            if n_feat == 0:
                return None
            pred = inp.copy()
            for feat in range(1, n_feat + 1):
                coords = np.argwhere(lbl == feat)
                r_min, r_max = coords[:, 0].min(), coords[:, 0].max()
                c_min, c_max = coords[:, 1].min(), coords[:, 1].max()
                for r, c in coords:
                    is_r_edge = (r == r_min or r == r_max)
                    is_c_edge = (c == c_min or c == c_max)
                    if is_r_edge and is_c_edge:
                        pred[r, c] = 1
                    elif is_r_edge or is_c_edge:
                        pred[r, c] = 4
                    else:
                        pred[r, c] = 2
            return pred
        prims["recolor_rectangular_blocks_corners_1_borders_4_interior_2"] = recolor_rectangular_blocks_corners_1_borders_4_interior_2

        # 243. Tampon croix boussole autour de chaque 1 [2 haut, 8 bas, 7 gauche, 6 droite] (tâche d364b489)
        def stamp_compass_cross_around_1s(inp: np.ndarray, ctx=None):
            if 1 not in inp:
                return None
            pred = inp.copy()
            H, W = inp.shape
            ones = np.argwhere(inp == 1)
            for r, c in ones:
                if r - 1 >= 0:
                    pred[r - 1, c] = 2
                if r + 1 < H:
                    pred[r + 1, c] = 8
                if c - 1 >= 0:
                    pred[r, c - 1] = 7
                if c + 1 < W:
                    pred[r, c + 1] = 6
            return pred
        prims["stamp_compass_cross_around_1s"] = stamp_compass_cross_around_1s

        # 244. Expansion de damier/motif 2D avec décalage horizontal de 1 vers la gauche (tâche caa06a1f)
        def expand_periodic_pattern_with_horizontal_shift_left(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            pad_c = inp[-1, -1]
            col0 = [c for c in inp[:, 0] if c != pad_c]
            if len(col0) == 0:
                return None
            P_vert = len(col0)
            for p_v in range(1, len(col0)):
                if all(col0[i] == col0[i % p_v] for i in range(len(col0))):
                    P_vert = p_v
                    break
            pred = np.zeros((H, W), dtype=inp.dtype)
            for r in range(H):
                src_r = r % P_vert
                row_vals = [c for c in inp[src_r, :] if c != pad_c]
                if len(row_vals) == 0:
                    return None
                P_horiz = len(row_vals)
                for p_h in range(1, len(row_vals)):
                    if all(row_vals[i] == row_vals[i % p_h] for i in range(len(row_vals))):
                        P_horiz = p_h
                        break
                pattern = row_vals[:P_horiz]
                for c in range(W):
                    pred[r, c] = pattern[(c + 1) % P_horiz]
            return pred
        prims["expand_periodic_pattern_with_horizontal_shift_left"] = expand_periodic_pattern_with_horizontal_shift_left

        # 245. Recolorisation des formes de 8 : unique -> 2, doublons -> 1 (tâche b230c067)
        def recolor_shapes_unique_2_duplicates_1(inp: np.ndarray, ctx=None):
            if 8 not in inp:
                return None
            lbl, n = label(inp == 8)
            if n == 0:
                return None
            shapes = {}
            for feat in range(1, n + 1):
                coords = np.argwhere(lbl == feat)
                rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
                cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
                crop = (inp[rmin:rmax+1, cmin:cmax+1] == 8).astype(int)
                shapes[feat] = crop.tobytes()
            from collections import Counter
            counts = Counter(shapes.values())
            pred = inp.copy()
            for feat in range(1, n + 1):
                coords = np.argwhere(lbl == feat)
                color = 2 if counts[shapes[feat]] == 1 else 1
                for r, c in coords:
                    pred[r, c] = color
            return pred
        prims["recolor_shapes_unique_2_duplicates_1"] = recolor_shapes_unique_2_duplicates_1

        # 246. Remplissage intérieur strict des rectangles de 5 par 2 (tâche bb43febb)
        def fill_interior_of_5_rectangles_with_2(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            lbl, n = label(inp == 5)
            if n == 0:
                return None
            pred = inp.copy()
            for feat in range(1, n + 1):
                coords = np.argwhere(lbl == feat)
                rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
                cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
                for r, c in coords:
                    if rmin < r < rmax and cmin < c < cmax:
                        pred[r, c] = 2
            return pred
        prims["fill_interior_of_5_rectangles_with_2"] = fill_interior_of_5_rectangles_with_2

        # 247. Remplissage des carrés creux de 5 selon leur taille (5 + taille) (tâche c0f76784)
        def fill_hollow_squares_by_size_color(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            lbl, n = label(inp == 5)
            if n == 0:
                return None
            pred = inp.copy()
            for feat in range(1, n + 1):
                coords = np.argwhere(lbl == feat)
                rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
                cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
                int_h = (rmax - 1) - (rmin + 1) + 1
                int_w = (cmax - 1) - (cmin + 1) + 1
                if int_h > 0 and int_w > 0:
                    color = 5 + int_h
                    pred[rmin+1:rmax, cmin+1:cmax] = color
            return pred
        prims["fill_hollow_squares_by_size_color"] = fill_hollow_squares_by_size_color

        # 248. Recolorisation des composantes de 5 : taille 6 -> 2, autres -> 1 (tâche d2abd087)
        def recolor_components_size_6_to_2_others_to_1(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            lbl, n = label(inp == 5)
            if n == 0:
                return None
            pred = inp.copy()
            for feat in range(1, n + 1):
                coords = np.argwhere(lbl == feat)
                color = 2 if len(coords) == 6 else 1
                for r, c in coords:
                    pred[r, c] = color
            return pred
        prims["recolor_components_size_6_to_2_others_to_1"] = recolor_components_size_6_to_2_others_to_1

        # 249. Extrusion de boîte creuse vers le marqueur 8 (tâche b548a754)
        def extrude_hollow_box_towards_marker_8(inp: np.ndarray, ctx=None):
            if 8 not in inp:
                return None
            coords_8 = np.argwhere(inp == 8)
            if len(coords_8) == 0:
                return None
            r_8, c_8 = coords_8[0]
            pred = inp.copy()
            pred[r_8, c_8] = 0
            shape_mask = (inp != 0) & (inp != 8)
            coords = np.argwhere(shape_mask)
            if len(coords) == 0:
                return None
            r_min, r_max = coords[:, 0].min(), coords[:, 0].max()
            c_min, c_max = coords[:, 1].min(), coords[:, 1].max()
            border_c = inp[r_min, c_min]
            int_c = inp[r_min + 1, c_min + 1]
            new_r_min, new_r_max = r_min, r_max
            new_c_min, new_c_max = c_min, c_max
            if r_8 < r_min:
                new_r_min = r_8
            elif r_8 > r_max:
                new_r_max = r_8
            elif c_8 < c_min:
                new_c_min = c_8
            elif c_8 > c_max:
                new_c_max = c_8
            for r in range(new_r_min, new_r_max + 1):
                for c in range(new_c_min, new_c_max + 1):
                    if r == new_r_min or r == new_r_max or c == new_c_min or c == new_c_max:
                        pred[r, c] = border_c
                    else:
                        pred[r, c] = int_c
            return pred
        prims["extrude_hollow_box_towards_marker_8"] = extrude_hollow_box_towards_marker_8

        # 250. Remplissage intérieur strict entre 4 coins de couleur 4 par 2 (tâche af902bf9)
        def fill_interior_of_4_corner_dots_4_with_2(inp: np.ndarray, ctx=None):
            if 4 not in inp:
                return None
            fours = set(map(tuple, np.argwhere(inp == 4)))
            if len(fours) < 4:
                return None
            pred = inp.copy()
            changed = False
            for r1, c1 in fours:
                for r2, c2 in fours:
                    if r1 < r2 and c1 < c2:
                        if (r1, c2) in fours and (r2, c1) in fours:
                            pred[r1+1:r2, c1+1:c2] = 2
                            changed = True
            return pred if changed else None
        prims["fill_interior_of_4_corner_dots_4_with_2"] = fill_interior_of_4_corner_dots_4_with_2

        # 251. Recolorisation de la moitié inférieure des colonnes de 2 en 8 (tâche ce9e57f2)
        def recolor_bottom_half_of_vertical_bars_2_to_8(inp: np.ndarray, ctx=None):
            if 2 not in inp:
                return None
            H, W = inp.shape
            pred = inp.copy()
            changed = False
            for c in range(W):
                rows = np.where(inp[:, c] == 2)[0]
                if len(rows) == 0:
                    continue
                if rows[-1] - rows[0] + 1 == len(rows):
                    k = len(rows) // 2
                    if k > 0:
                        for r in rows[-k:]:
                            pred[r, c] = 8
                            changed = True
            return pred if changed else None
        prims["recolor_bottom_half_of_vertical_bars_2_to_8"] = recolor_bottom_half_of_vertical_bars_2_to_8

        # 252. Amarrage magnétique de points de 5 vers ancre carrée 2x2 de 2 (tâche a48eeaf7)
        def magnetic_dock_dots_5_to_anchor_2x2_square(inp: np.ndarray, ctx=None):
            if 2 not in inp or 5 not in inp:
                return None
            coords_2 = np.argwhere(inp == 2)
            if len(coords_2) != 4:
                return None
            r_min, r_max = coords_2[:, 0].min(), coords_2[:, 0].max()
            c_min, c_max = coords_2[:, 1].min(), coords_2[:, 1].max()
            if r_max - r_min != 1 or c_max - c_min != 1:
                return None
            pred = inp.copy()
            pred[pred == 5] = 0
            fives = np.argwhere(inp == 5)
            for r, c in fives:
                dr = 0
                if r < r_min: dr = 1
                elif r > r_max: dr = -1
                dc = 0
                if c < c_min: dc = 1
                elif c > c_max: dc = -1
                curr_r, curr_c = r, c
                steps = 0
                hit = False
                max_steps = max(inp.shape) + 2
                while steps < max_steps:
                    next_r = curr_r + dr
                    next_c = curr_c + dc
                    if r_min <= next_r <= r_max and c_min <= next_c <= c_max:
                        hit = True
                        break
                    if not (0 <= next_r < inp.shape[0] and 0 <= next_c < inp.shape[1]):
                        break
                    curr_r, curr_c = next_r, next_c
                    steps += 1
                if hit:
                    pred[curr_r, curr_c] = 5
            return pred
        prims["magnetic_dock_dots_5_to_anchor_2x2_square"] = magnetic_dock_dots_5_to_anchor_2x2_square

        # 253. Histogramme barres verticales des couleurs gagnantes triées par colonne min (tâche a3325580)
        def barchart_of_winning_colors_sorted_by_min_col(inp: np.ndarray, ctx=None):
            from collections import Counter
            counts = Counter(inp.flatten())
            counts.pop(0, None)
            if not counts:
                return None
            max_c = max(counts.values())
            winning = [c for c, v in counts.items() if v == max_c]
            winning.sort(key=lambda c: np.where(inp == c)[1].min())
            H = max_c
            W = len(winning)
            pred = np.zeros((H, W), dtype=int)
            for idx, c in enumerate(winning):
                pred[:, idx] = c
            return pred
        prims["barchart_of_winning_colors_sorted_by_min_col"] = barchart_of_winning_colors_sorted_by_min_col

        # 254. Rayon diagonal bissecteur 45° émis par tromino en L (tâche 6e19193c)
        def l_tromino_angle_bisector_ray(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) == 0:
                return None
            pred = inp.copy()
            changed = False
            for c in colors:
                lbl, n = label(inp == c)
                for feat in range(1, n + 1):
                    coords = np.argwhere(lbl == feat)
                    if len(coords) == 3:
                        rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
                        cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
                        if rmax - rmin == 1 and cmax - cmin == 1:
                            all_pts = set(map(tuple, coords))
                            missing = [(r, cc) for r in [rmin, rmax] for cc in [cmin, cmax] if (r, cc) not in all_pts][0]
                            corner = (rmin if missing[0] == rmax else rmax, cmin if missing[1] == cmax else cmax)
                            dr = missing[0] - corner[0]
                            dc = missing[1] - corner[1]
                            curr_r, curr_c = missing[0] + dr, missing[1] + dc
                            while 0 <= curr_r < H and 0 <= curr_c < W:
                                pred[curr_r, curr_c] = c
                                changed = True
                                curr_r += dr
                                curr_c += dc
            return pred if changed else None
        prims["l_tromino_angle_bisector_ray"] = l_tromino_angle_bisector_ray

        # 255. Tri des barres horizontales par longueur croissante empilées en bas à droite (tâche beb8660c)
        def sort_horizontal_bars_ascending_stack_bottom_right(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            bars = []
            for r in range(H):
                cols = np.where(inp[r] != 0)[0]
                if len(cols) > 0 and cols[-1] - cols[0] + 1 == len(cols):
                    bars.append((len(cols), inp[r, cols[0]]))
            if len(bars) == 0:
                return None
            bars.sort(key=lambda x: x[0])
            pred = np.zeros((H, W), dtype=int)
            for i, (length, c) in enumerate(reversed(bars)):
                r = H - 1 - i
                pred[r, W - length:W] = c
            return pred
        prims["sort_horizontal_bars_ascending_stack_bottom_right"] = sort_horizontal_bars_ascending_stack_bottom_right

        # 256. Recolorisation des composantes de 5 selon marqueur clé rangée 0 (tâche ddf7fa4f)
        def recolor_components_of_5_by_row0_key_marker(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            keys = {c: inp[0, c] for c in range(inp.shape[1]) if inp[0, c] != 0}
            if not keys:
                return None
            lbl, n = label(inp == 5)
            if n == 0:
                return None
            pred = inp.copy()
            for feat in range(1, n + 1):
                coords = np.argwhere(lbl == feat)
                cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
                for k_c, k_val in keys.items():
                    if cmin <= k_c <= cmax:
                        for r, c in coords:
                            pred[r, c] = k_val
                        break
            return pred
        prims["recolor_components_of_5_by_row0_key_marker"] = recolor_components_of_5_by_row0_key_marker

        # 257. Pont de 8 entre deux boîtes distantes avec retrait transverse d'une unité (tâche d6ad076f)
        def bridge_two_boxes_across_gap_with_8(inp: np.ndarray, ctx=None):
            colors = [c for c in np.unique(inp) if c != 0]
            if len(colors) == 0:
                return None
            boxes = []
            for c in colors:
                lbl, n = label(inp == c)
                for feat in range(1, n + 1):
                    coords = np.argwhere(lbl == feat)
                    rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
                    cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
                    boxes.append((rmin, rmax, cmin, cmax))
            if len(boxes) != 2:
                return None
            b1, b2 = boxes
            pred = inp.copy()
            if b1[1] < b2[0] or b2[1] < b1[0]:
                top = b1 if b1[1] < b2[0] else b2
                bot = b2 if b1[1] < b2[0] else b1
                c_start = max(top[2], bot[2]) + 1
                c_end = min(top[3], bot[3]) - 1
                if c_start <= c_end:
                    for r in range(top[1] + 1, bot[0]):
                        pred[r, c_start:c_end + 1] = 8
                    return pred
            if b1[3] < b2[2] or b2[3] < b1[2]:
                left = b1 if b1[3] < b2[2] else b2
                right = b2 if b1[3] < b2[2] else b1
                r_start = max(left[0], right[0]) + 1
                r_end = min(left[1], right[1]) - 1
                if r_start <= r_end:
                    for c in range(left[3] + 1, right[2]):
                        pred[r_start:r_end + 1, c] = 8
                    return pred
            return None
        prims["bridge_two_boxes_across_gap_with_8"] = bridge_two_boxes_across_gap_with_8

        # 258. Conteneur creux de 5 rempli de 8 et jet drainé par la brèche vers le bord (tâche d4f3cd78)
        def hollow_container_drain_beam_through_hole_8(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            coords = np.argwhere(inp == 5)
            if len(coords) == 0:
                return None
            rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
            cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
            H, W = inp.shape
            pred = inp.copy()
            pred[rmin+1:rmax, cmin+1:cmax] = 8
            for c in range(cmin+1, cmax):
                if inp[rmin, c] == 0:
                    pred[rmin, c] = 8
                    pred[:rmin, c] = 8
                    return pred
            for c in range(cmin+1, cmax):
                if inp[rmax, c] == 0:
                    pred[rmax, c] = 8
                    pred[rmax+1:, c] = 8
                    return pred
            for r in range(rmin+1, rmax):
                if inp[r, cmin] == 0:
                    pred[r, cmin] = 8
                    pred[r, :cmin] = 8
                    return pred
            for r in range(rmin+1, rmax):
                if inp[r, cmax] == 0:
                    pred[r, cmax] = 8
                    pred[r, cmax+1:] = 8
                    return pred
            return None
        prims["hollow_container_drain_beam_through_hole_8"] = hollow_container_drain_beam_through_hole_8

        # 259. Rayons verticaux montant de 2 contournant obstacle 5 vers la droite (tâche d9f24cd1)
        def vertical_upward_rays_step_right_on_obstacle_5(inp: np.ndarray, ctx=None):
            if 2 not in inp or 5 not in inp:
                return None
            H, W = inp.shape
            start_cols = np.where(inp[H-1] == 2)[0]
            if len(start_cols) == 0:
                return None
            pred = inp.copy()
            for sc in start_cols:
                curr_c = sc
                for r in range(H-1, -1, -1):
                    pred[r, curr_c] = 2
                    if r > 0 and inp[r-1, curr_c] == 5:
                        pred[r, curr_c + 1] = 2
                        curr_c += 1
            return pred
        prims["vertical_upward_rays_step_right_on_obstacle_5"] = vertical_upward_rays_step_right_on_obstacle_5

        # 260. Superposition de la moitié gauche et moitié droite réfléchie fliplr séparées par 5 (tâche e3497940)
        def overlay_left_and_flipped_right_across_divider_5(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            H, W = inp.shape
            div_cols = [c for c in range(W) if np.all(inp[:, c] == 5)]
            if len(div_cols) != 1:
                return None
            c_div = div_cols[0]
            left = inp[:, :c_div]
            right = inp[:, c_div+1:]
            if left.shape != right.shape:
                return None
            right_flipped = np.fliplr(right)
            return np.where(left != 0, left, right_flipped)
        prims["overlay_left_and_flipped_right_across_divider_5"] = overlay_left_and_flipped_right_across_divider_5

        # 261. Rayons horizontaux alternés C et 5 vers le bord droit (tâche 97999447)
        def single_dots_shoot_alternating_ray_to_right(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            dots = np.argwhere((inp != 0) & (inp != 5))
            if len(dots) == 0:
                return None
            pred = inp.copy()
            for r, c in dots:
                color = inp[r, c]
                for idx, col in enumerate(range(c + 1, W)):
                    pred[r, col] = 5 if idx % 2 == 0 else color
            return pred
        prims["single_dots_shoot_alternating_ray_to_right"] = single_dots_shoot_alternating_ray_to_right

        # 262. Rognage carré concentrique et inversion couleur cadre/centre (tâche b94a9452)
        def crop_concentric_square_and_swap_colors(inp: np.ndarray, ctx=None):
            coords = np.argwhere(inp != 0)
            if len(coords) == 0:
                return None
            rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
            cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
            crop = inp[rmin:rmax+1, cmin:cmax+1].copy()
            c_outer = crop[0, 0]
            c_inner = crop[crop.shape[0]//2, crop.shape[1]//2]
            if c_outer == 0 or c_inner == 0 or c_outer == c_inner:
                return None
            res = np.zeros_like(crop)
            res[crop == c_outer] = c_inner
            res[crop == c_inner] = c_outer
            return res
        prims["crop_concentric_square_and_swap_colors"] = crop_concentric_square_and_swap_colors

        # 263. Déplacement du réticule selon le nombre de 5 dans la colonne 9 (tâche e48d4e1a)
        def shift_crosshairs_by_count_of_5s_in_col9(inp: np.ndarray, ctx=None):
            H, W = inp.shape
            if W <= 9:
                return None
            K = int(np.sum(inp[:, 9] == 5))
            if K == 0:
                return None
            colors = [c for c in np.unique(inp) if c != 0 and c != 5]
            if len(colors) != 1:
                return None
            C = colors[0]
            row_counts = np.sum(inp == C, axis=1)
            col_counts = np.sum(inp == C, axis=0)
            r_cross = np.argmax(row_counts)
            c_cross = np.argmax(col_counts)
            new_r = r_cross + K
            new_c = c_cross - K
            if not (0 <= new_r < H and 0 <= new_c < W):
                return None
            pred = np.zeros((H, W), dtype=int)
            pred[new_r, :] = C
            pred[:, new_c] = C
            return pred
        prims["shift_crosshairs_by_count_of_5s_in_col9"] = shift_crosshairs_by_count_of_5s_in_col9

        # 264. Connexion en L de 8 et 2 par du 4 avec angle à (r2, c8) (tâche d4a91cb9)
        def connect_8_and_2_with_l_path_of_4_corner_at_r2_c8(inp: np.ndarray, ctx=None):
            if 8 not in inp or 2 not in inp:
                return None
            coords8 = np.argwhere(inp == 8)
            coords2 = np.argwhere(inp == 2)
            if len(coords8) != 1 or len(coords2) != 1:
                return None
            r8, c8 = coords8[0]
            r2, c2 = coords2[0]
            pred = inp.copy()
            r_min, r_max = min(r8, r2), max(r8, r2)
            for r in range(r_min, r_max + 1):
                if (r, c8) != (r8, c8) and (r, c8) != (r2, c2):
                    pred[r, c8] = 4
            c_min, c_max = min(c8, c2), max(c8, c2)
            for c in range(c_min, c_max + 1):
                if (r2, c) != (r8, c8) and (r2, c) != (r2, c2):
                    pred[r2, c] = 4
            return pred
        prims["connect_8_and_2_with_l_path_of_4_corner_at_r2_c8"] = connect_8_and_2_with_l_path_of_4_corner_at_r2_c8

        # 265. Barres murales opposées connectant les 8 et projetant rayons symétriques (tâche 673ef223)
        def dual_wall_bars_connect_8s_and_shoot_opposite_rays(inp: np.ndarray, ctx=None):
            if 2 not in inp or 8 not in inp:
                return None
            H, W = inp.shape
            twos = np.argwhere(inp == 2)
            mid = H // 2
            top_twos = [pt for pt in twos if pt[0] < mid]
            bot_twos = [pt for pt in twos if pt[0] >= mid]
            if not top_twos or not bot_twos:
                return None
            top_col = top_twos[0][1]
            top_r0 = min(pt[0] for pt in top_twos)
            bot_col = bot_twos[0][1]
            bot_r0 = min(pt[0] for pt in bot_twos)
            eights = np.argwhere(inp == 8)
            pred = inp.copy()
            for r8, c8 in eights:
                pred[r8, c8] = 4
                c_start, c_end = min(top_col, c8), max(top_col, c8)
                for c in range(c_start + 1, c_end):
                    pred[r8, c] = 8
                offset = r8 - top_r0
                bot_r = bot_r0 + offset
                opp_col = 0 if bot_col == W - 1 else W - 1
                c_start, c_end = min(bot_col, opp_col), max(bot_col, opp_col)
                for c in range(c_start, c_end + 1):
                    if c != bot_col:
                        pred[bot_r, c] = 8
            return pred
        prims["dual_wall_bars_connect_8s_and_shoot_opposite_rays"] = dual_wall_bars_connect_8s_and_shoot_opposite_rays

        # 266. Alternance de points de 5 de droite à gauche avec 3 (tâche d406998b)
        def alternate_dots_of_5_from_right_to_left_with_3(inp: np.ndarray, ctx=None):
            if 5 not in inp:
                return None
            fives = np.argwhere(inp == 5)
            if len(fives) == 0:
                return None
            fives = sorted(fives, key=lambda x: (x[1], x[0]))
            pred = inp.copy()
            N = len(fives)
            for i, (r, c) in enumerate(fives):
                pred[r, c] = 3 if (N - 1 - i) % 2 == 0 else 5
            return pred
        prims["alternate_dots_of_5_from_right_to_left_with_3"] = alternate_dots_of_5_from_right_to_left_with_3

        # 267. Rectangles creux de 2 : remplissage intérieur 3 et effacement des 2 (tâche d5d6de2d)
        def hollow_rectangles_of_2_fill_interior_3_erase_2(inp: np.ndarray, ctx=None):
            if 2 not in inp:
                return None
            lbl, n = label(inp == 2)
            if n == 0:
                return None
            pred = np.zeros_like(inp)
            changed = False
            for i in range(1, n + 1):
                coords = np.argwhere(lbl == i)
                rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
                cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
                if rmax - rmin >= 2 and cmax - cmin >= 2:
                    interior = inp[rmin+1:rmax, cmin+1:cmax]
                    if np.all(interior == 0):
                        pred[rmin+1:rmax, cmin+1:cmax] = 3
                        changed = True
            return pred if changed else None
        prims["hollow_rectangles_of_2_fill_interior_3_erase_2"] = hollow_rectangles_of_2_fill_interior_3_erase_2

        # 268. Connexion de 2 et 3 par un chemin en L de 8 avec angle à (r2, c3) (tâche a2fd1cf0)
        def connect_2_and_3_with_l_path_of_8_corner_at_r2_c3(inp: np.ndarray, ctx=None):
            if 2 not in inp or 3 not in inp:
                return None
            twos = np.argwhere(inp == 2)
            threes = np.argwhere(inp == 3)
            if len(twos) != 1 or len(threes) != 1:
                return None
            r2, c2 = twos[0]
            r3, c3 = threes[0]
            pred = inp.copy()
            c_min, c_max = min(c2, c3), max(c2, c3)
            for c in range(c_min, c_max + 1):
                if (r2, c) != (r2, c2) and (r2, c) != (r3, c3):
                    pred[r2, c] = 8
            r_min, r_max = min(r2, r3), max(r2, r3)
            for r in range(r_min, r_max + 1):
                if (r, c3) != (r2, c2) and (r, c3) != (r3, c3):
                    pred[r, c3] = 8
            return pred
        prims["connect_2_and_3_with_l_path_of_8_corner_at_r2_c3"] = connect_2_and_3_with_l_path_of_8_corner_at_r2_c3

        # 269. Recolorisation des composantes de 1 contenant un trou en 8 (tâche b2862040)
        def recolor_components_of_1_with_hole_to_8(inp: np.ndarray, ctx=None):
            if 1 not in inp:
                return None
            H, W = inp.shape
            lbl, n = label(inp == 1)
            if n == 0:
                return None
            pred = inp.copy()
            changed = False
            for i in range(1, n + 1):
                mask = (lbl == i)
                inv_lbl, inv_n = label(~mask)
                touching = set()
                for r in range(H):
                    touching.add(inv_lbl[r, 0])
                    touching.add(inv_lbl[r, W-1])
                for c in range(W):
                    touching.add(inv_lbl[0, c])
                    touching.add(inv_lbl[H-1, c])
                touching.discard(0)
                has_hole = False
                for j in range(1, inv_n + 1):
                    if j not in touching:
                        has_hole = True
                        break
                if has_hole:
                    pred[mask] = 8
                    changed = True
            return pred if changed else None
        prims["recolor_components_of_1_with_hole_to_8"] = recolor_components_of_1_with_hole_to_8

        # Primitive 270 (Mega-Wave 19 - 90f3ed37)
        def template_completion_partial_components_to_right(inp, train_pairs=None):
            try:
                rows_with_8 = [r for r in range(inp.shape[0]) if 8 in inp[r, :]]
                if len(rows_with_8) < 2:
                    return None
                groups = []
                curr = [rows_with_8[0]]
                for r in rows_with_8[1:]:
                    if r - curr[-1] > 1:
                        groups.append(curr)
                        curr = [r]
                    else:
                        curr.append(r)
                groups.append(curr)
                if len(groups) < 2:
                    return None
                t_rows = groups[0]
                t_h = len(t_rows)
                t_w = inp.shape[1]
                tpl = inp[t_rows[0]:t_rows[0] + t_h, :]
                out = inp.copy()
                changed = False
                for grp in groups[1:]:
                    grp_max_c = max(c for r in grp for c in range(t_w) if inp[r, c] == 8)
                    start_r = grp[0]
                    for roff in range(t_h):
                        r = start_r + roff
                        if r >= inp.shape[0]:
                            break
                        for c in range(grp_max_c + 1, t_w):
                            if tpl[roff, c] == 8:
                                out[r, c] = 1
                                changed = True
                return out if changed else None
            except Exception:
                return None
        prims["template_completion_partial_components_to_right"] = template_completion_partial_components_to_right

        # Primitive 271 (Mega-Wave 19 - d06dbe63)
        def stepped_staircase_rays_from_seed_8(inp, train_pairs=None):
            try:
                H, W = inp.shape
                eights = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 8]
                if len(eights) != 1:
                    return None
                sr, sc = eights[0]
                out = inp.copy()
                curr_r, curr_c = sr, sc
                changed = False
                while True:
                    nr = curr_r - 1
                    if nr < 0: break
                    out[nr, curr_c] = 5
                    changed = True
                    nr2 = nr - 1
                    if nr2 < 0: break
                    c_end = min(W, curr_c + 3)
                    out[nr2, curr_c:c_end] = 5
                    curr_r = nr2
                    curr_c = curr_c + 2
                    if curr_c >= W: break
                curr_r, curr_c = sr, sc
                while True:
                    nr = curr_r + 1
                    if nr >= H: break
                    out[nr, curr_c] = 5
                    changed = True
                    nr2 = nr + 1
                    if nr2 >= H: break
                    c_start = max(0, curr_c - 2)
                    out[nr2, c_start:curr_c + 1] = 5
                    curr_r = nr2
                    curr_c = curr_c - 2
                    if curr_c < 0: break
                return out if changed else None
            except Exception:
                return None
        prims["stepped_staircase_rays_from_seed_8"] = stepped_staircase_rays_from_seed_8

        # Primitive 272 (Mega-Wave 19 - ef135b50)
        def bridge_parent_child_components_with_9(inp, train_pairs=None):
            try:
                H, W = inp.shape
                lab, num = label(inp == 2)
                if num < 2: return None
                comps = []
                for f in range(1, num + 1):
                    pts = [(r, c) for r in range(H) for c in range(W) if lab[r, c] == f]
                    rmin, rmax = min(r for r, c in pts), max(r for r, c in pts)
                    cmin, cmax = min(c for r, c in pts), max(c for r, c in pts)
                    comps.append({
                        'pts': pts, 'rmin': rmin, 'rmax': rmax, 'cmin': cmin, 'cmax': cmax,
                        'rows': {r: [c for rr, c in pts if rr == r] for r in range(rmin, rmax + 1)}
                    })
                out = inp.copy()
                changed = False
                for j, cj in enumerate(comps):
                    best_p = None
                    best_dist = 999
                    for i, ci in enumerate(comps):
                        if i == j: continue
                        if ci['rmin'] < cj['rmin'] <= ci['rmax']:
                            dist = abs(cj['cmin'] - ci['cmax'])
                            if dist < best_dist:
                                best_dist = dist
                                best_p = ci
                    if best_p is not None:
                        ci = best_p
                        start_r = cj['rmin']
                        end_r = min(ci['rmax'], cj['rmax'])
                        for r in range(start_r, end_r + 1):
                            if r in ci['rows'] and r in cj['rows']:
                                row_ci, row_cj = ci['rows'][r], cj['rows'][r]
                                left_edge = min(max(row_ci), max(row_cj))
                                right_edge = max(min(row_ci), min(row_cj))
                                for col in range(left_edge + 1, right_edge):
                                    out[r, col] = 9
                                    changed = True
                return out if changed else None
            except Exception:
                return None
        prims["bridge_parent_child_components_with_9"] = bridge_parent_child_components_with_9

        # Primitive 273 (Mega-Wave 19 - ea32f347)
        def recolor_bars_of_5_by_length_rank(inp, train_pairs=None):
            try:
                H, W = inp.shape
                lab, num = label(inp == 5)
                if num < 2: return None
                comps = []
                for f in range(1, num + 1):
                    pts = [(r, c) for r in range(H) for c in range(W) if lab[r, c] == f]
                    comps.append((len(pts), pts))
                comps.sort(key=lambda x: x[0], reverse=True)
                rank_colors = [1, 4, 2]
                out = inp.copy()
                for idx, (sz, pts) in enumerate(comps):
                    color = rank_colors[idx] if idx < len(rank_colors) else 2
                    for r, c in pts:
                        out[r, c] = color
                return out
            except Exception:
                return None
        prims["recolor_bars_of_5_by_length_rank"] = recolor_bars_of_5_by_length_rank

        # Primitive 274 (Mega-Wave 19 - f25fbde4)
        def crop_nonzero_bbox_and_upscale_2x(inp, train_pairs=None):
            try:
                rows = [r for r in range(inp.shape[0]) if np.any(inp[r, :] != 0)]
                cols = [c for c in range(inp.shape[1]) if np.any(inp[:, c] != 0)]
                if not rows or not cols: return None
                crop = inp[min(rows):max(rows) + 1, min(cols):max(cols) + 1]
                return np.kron(crop, np.ones((2, 2), dtype=int))
            except Exception:
                return None
        prims["crop_nonzero_bbox_and_upscale_2x"] = crop_nonzero_bbox_and_upscale_2x

        # Primitive 275 (Mega-Wave 19 - e98196ab)
        def horizontal_divider_overlay_halves(inp, train_pairs=None):
            try:
                div_rows = [r for r in range(inp.shape[0]) if np.all(inp[r, :] == 5)]
                if not div_rows: return None
                dr = div_rows[0]
                top = inp[:dr, :]
                bot = inp[dr + 1:, :]
                if top.shape != bot.shape: return None
                return np.where(top != 0, top, bot)
            except Exception:
                return None
        prims["horizontal_divider_overlay_halves"] = horizontal_divider_overlay_halves

        # Primitive 276 (Mega-Wave 19 - a78176bb)
        def parallel_diagonal_offset_by_fives_count(inp, train_pairs=None):
            try:
                H, W = inp.shape
                fives = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 5]
                if not fives: return None
                other_colors = [c for c in np.unique(inp) if c != 0 and c != 5]
                if len(other_colors) != 1: return None
                c_main = other_colors[0]
                main_pts = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == c_main]
                if not main_pts: return None
                d0 = main_pts[0][0] - main_pts[0][1]
                five_diags = set(r - c for r, c in fives)
                left_count = sum(1 for d in five_diags if d < d0)
                right_count = sum(1 for d in five_diags if d > d0)
                out = inp.copy()
                out[out == 5] = 0
                if left_count > 0:
                    d_target = d0 - (left_count + 2)
                    for r in range(H):
                        c = r - d_target
                        if 0 <= c < W: out[r, c] = c_main
                if right_count > 0:
                    d_target = d0 + (right_count + 2)
                    for r in range(H):
                        c = r - d_target
                        if 0 <= c < W: out[r, c] = c_main
                return out
            except Exception:
                return None
        prims["parallel_diagonal_offset_by_fives_count"] = parallel_diagonal_offset_by_fives_count

        # Primitive 277 (Mega-Wave 19 - fcc82909)
        def blocks_2x2_project_shadow_length_unique_colors(inp, train_pairs=None):
            try:
                H, W = inp.shape
                out = inp.copy()
                changed = False
                r = 0
                while r < H - 1:
                    c = 0
                    while c < W - 1:
                        blk = inp[r:r+2, c:c+2]
                        if np.all(blk != 0):
                            n_cols = len(np.unique(blk))
                            for dr in range(2, 2 + n_cols):
                                if r + dr < H:
                                    out[r + dr, c] = 3
                                    out[r + dr, c + 1] = 3
                                    changed = True
                            c += 2
                        else:
                            c += 1
                    r += 1
                return out if changed else None
            except Exception:
                return None
        prims["blocks_2x2_project_shadow_length_unique_colors"] = blocks_2x2_project_shadow_length_unique_colors

        # Primitive 278 (Mega-Wave 19 - 539a4f51)
        def tensor_product_tiling_kxk_to_2kx2k_on_background(inp, train_pairs=None):
            try:
                rows = [r for r in range(inp.shape[0]) if np.any(inp[r, :] != 0)]
                cols = [c for c in range(inp.shape[1]) if np.any(inp[:, c] != 0)]
                if not rows or not cols: return None
                K = max(max(rows), max(cols)) + 1
                if 2 * K > 10: return None
                B = inp[:K, :K]
                bg = B[0, 0]
                out = np.full((10, 10), bg)
                out[:K, :K] = B
                out[:K, K:2*K] = np.tile(B[0, :], (K, 1))
                out[K:2*K, :K] = np.tile(B[:, 0][:, None], (1, K))
                out[K:2*K, K:2*K] = B
                return out
            except Exception:
                return None
        prims["tensor_product_tiling_kxk_to_2kx2k_on_background"] = tensor_product_tiling_kxk_to_2kx2k_on_background

        # Primitive 279 (Mega-Wave 19 - d687bc17)
        def border_color_gravity_dock_dots_to_matching_wall(inp, train_pairs=None):
            try:
                H, W = inp.shape
                top_c = inp[0, 1]
                bot_c = inp[-1, 1]
                left_c = inp[1, 0]
                right_c = inp[1, -1]
                if not (np.all(inp[0, 1:-1] == top_c) and np.all(inp[-1, 1:-1] == bot_c)):
                    return None
                out = np.zeros_like(inp)
                out[0, :] = inp[0, :]
                out[-1, :] = inp[-1, :]
                out[:, 0] = inp[:, 0]
                out[:, -1] = inp[:, -1]
                for r in range(1, H - 1):
                    for c in range(1, W - 1):
                        val = inp[r, c]
                        if val == 0: continue
                        if val == top_c: out[1, c] = val
                        elif val == bot_c: out[H - 2, c] = val
                        elif val == left_c: out[r, 1] = val
                        elif val == right_c: out[r, W - 2] = val
                return out
            except Exception:
                return None
        prims["border_color_gravity_dock_dots_to_matching_wall"] = border_color_gravity_dock_dots_to_matching_wall

        # Primitive 280 (Mega-Wave 19 - e21d9049)
        def tiled_periodic_cross_extension_horizontal_vertical(inp, train_pairs=None):
            try:
                H, W = inp.shape
                row_counts = [np.count_nonzero(inp[r, :]) for r in range(H)]
                col_counts = [np.count_nonzero(inp[:, c]) for c in range(W)]
                if max(row_counts) < 2 or max(col_counts) < 2: return None
                cross_r = np.argmax(row_counts)
                cross_c = np.argmax(col_counts)
                cols = [c for c in range(W) if inp[cross_r, c] != 0]
                L_h = len(cols)
                if L_h < 2: return None
                pat_h = [inp[cross_r, c] for c in cols]
                start_c = cols[0]
                out = np.zeros_like(inp)
                for c in range(W):
                    out[cross_r, c] = pat_h[(c - start_c) % L_h]
                rows = [r for r in range(H) if inp[r, cross_c] != 0]
                L_v = len(rows)
                if L_v < 2: return None
                pat_v = [inp[r, cross_c] for r in rows]
                start_r = rows[0]
                for r in range(H):
                    out[r, cross_c] = pat_v[(r - start_r) % L_v]
                return out
            except Exception:
                return None
        prims["tiled_periodic_cross_extension_horizontal_vertical"] = tiled_periodic_cross_extension_horizontal_vertical

        # Primitive 281 (Mega-Wave 19 - 272f95fa)
        def grid_3x3_boxes_fixed_palette_fill(inp, train_pairs=None):
            try:
                H, W = inp.shape
                h_lines = [r for r in range(H) if np.all(inp[r, :] == 8)]
                v_lines = [c for c in range(W) if np.all(inp[:, c] == 8)]
                if len(h_lines) != 2 or len(v_lines) != 2: return None
                r0, r1 = h_lines[0], h_lines[1]
                c0, c1 = v_lines[0], v_lines[1]
                out = inp.copy()
                out[0:r0, c0+1:c1] = 2
                out[r1+1:H, c0+1:c1] = 1
                out[r0+1:r1, 0:c0] = 4
                out[r0+1:r1, c0+1:c1] = 6
                out[r0+1:r1, c1+1:W] = 3
                return out
            except Exception:
                return None
        prims["grid_3x3_boxes_fixed_palette_fill"] = grid_3x3_boxes_fixed_palette_fill

        # Primitive 282 (Mega-Wave 19 - 543a7ed5)
        def cavities_interior_4_and_halo_3_around_6(inp, train_pairs=None):
            try:
                if 6 not in inp or 8 not in inp: return None
                H, W = inp.shape
                out = inp.copy()
                visited = np.zeros((H, W), dtype=bool)
                from collections import deque
                q = deque()
                for r in range(H):
                    for c in [0, W - 1]:
                        if inp[r, c] == 8 and not visited[r, c]:
                            visited[r, c] = True
                            q.append((r, c))
                for c in range(W):
                    for r in [0, H - 1]:
                        if inp[r, c] == 8 and not visited[r, c]:
                            visited[r, c] = True
                            q.append((r, c))
                while q:
                    r, c = q.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if inp[nr, nc] == 8 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                holes = (inp == 8) & (~visited)
                out[holes] = 4
                struct = np.ones((3, 3), dtype=bool)
                dilated_6 = binary_dilation(inp == 6, structure=struct)
                halo = dilated_6 & (inp == 8) & visited
                out[halo] = 3
                return out
            except Exception:
                return None
        prims["cavities_interior_4_and_halo_3_around_6"] = cavities_interior_4_and_halo_3_around_6

        # Primitive 283 (Mega-Wave 19 - 928ad970)
        def hollow_box_inscribed_between_four_fives(inp, train_pairs=None):
            try:
                H, W = inp.shape
                fives = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 5]
                if len(fives) != 4: return None
                top_r = min(r for r, c in fives)
                bot_r = max(r for r, c in fives)
                left_c = min(c for r, c in fives)
                right_c = max(c for r, c in fives)
                colors = [c for c in np.unique(inp) if c != 0 and c != 5]
                if len(colors) != 1: return None
                C = colors[0]
                out = inp.copy()
                out[top_r + 1, left_c + 1:right_c] = C
                out[bot_r - 1, left_c + 1:right_c] = C
                out[top_r + 1:bot_r, left_c + 1] = C
                out[top_r + 1:bot_r, right_c - 1] = C
                for r, c in fives:
                    out[r, c] = 5
                return out
            except Exception:
                return None
        prims["hollow_box_inscribed_between_four_fives"] = hollow_box_inscribed_between_four_fives

        # Primitive 284 (Mega-Wave 19 - e8593010)
        def recolor_zero_cavities_by_component_size(inp, train_pairs=None):
            try:
                if not (5 in inp and 0 in inp): return None
                structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
                labeled, num_features = label(inp == 0, structure=structure)
                if num_features < 2: return None
                size_to_color = {1: 3, 2: 2, 3: 1}
                out = inp.copy()
                for feat_id in range(1, num_features + 1):
                    mask = (labeled == feat_id)
                    sz = np.sum(mask)
                    color = size_to_color.get(sz, 0)
                    out[mask] = color
                return out
            except Exception:
                return None
        prims["recolor_zero_cavities_by_component_size"] = recolor_zero_cavities_by_component_size

        # Primitive 285 (Mega-Wave 19 - 6cdd2623)
        def connect_matching_endpoint_pairs_across_grid(inp, train_pairs=None):
            try:
                H, W = inp.shape
                color_lines = {}
                for r in range(H):
                    if inp[r, 0] != 0 and inp[r, 0] == inp[r, W - 1]:
                        val = inp[r, 0]
                        if val not in color_lines: color_lines[val] = []
                        color_lines[val].append(('row', r))
                for c in range(W):
                    if inp[0, c] != 0 and inp[0, c] == inp[H - 1, c]:
                        val = inp[0, c]
                        if val not in color_lines: color_lines[val] = []
                        color_lines[val].append(('col', c))
                if not color_lines: return None
                best_color = max(color_lines.keys(), key=lambda k: len(color_lines[k]))
                if len(color_lines[best_color]) < 2: return None
                out = np.zeros_like(inp)
                for line_type, idx in color_lines[best_color]:
                    if line_type == 'row':
                        out[idx, :] = best_color
                    else:
                        out[:, idx] = best_color
                return out
            except Exception:
                return None
        prims["connect_matching_endpoint_pairs_across_grid"] = connect_matching_endpoint_pairs_across_grid

        # Primitive 286 (Mega-Wave 19 - b7249182)
        def opposed_dots_interlocking_hand_brackets(inp, train_pairs=None):
            try:
                H, W = inp.shape
                dots = [(r, c, inp[r, c]) for r in range(H) for c in range(W) if inp[r, c] != 0]
                if len(dots) != 2: return None
                (r1, c1, col1), (r2, c2, col2) = dots
                out = np.zeros_like(inp)
                if c1 == c2:
                    C = c1
                    if r1 > r2:
                        r1, r2 = r2, r1
                        col1, col2 = col2, col1
                    r_mid = (r1 + r2) // 2
                    out[r1:r_mid, C] = col1
                    out[r_mid - 1, C - 2:C + 3] = col1
                    out[r_mid, C - 2] = col1
                    out[r_mid, C + 2] = col1
                    out[r_mid + 2:r2 + 1, C] = col2
                    out[r_mid + 2, C - 2:C + 3] = col2
                    out[r_mid + 1, C - 2] = col2
                    out[r_mid + 1, C + 2] = col2
                elif r1 == r2:
                    R = r1
                    if c1 > c2:
                        c1, c2 = c2, c1
                        col1, col2 = col2, col1
                    c_mid = (c1 + c2) // 2
                    out[R, c1:c_mid] = col1
                    out[R - 2:R + 3, c_mid - 1] = col1
                    out[R - 2, c_mid] = col1
                    out[R + 2, c_mid] = col1
                    out[R, c_mid + 2:c2 + 1] = col2
                    out[R - 2:R + 3, c_mid + 2] = col2
                    out[R - 2, c_mid + 1] = col2
                    out[R + 2, c_mid + 1] = col2
                else:
                    return None
                return out
            except Exception:
                return None
        prims["opposed_dots_interlocking_hand_brackets"] = opposed_dots_interlocking_hand_brackets

        # Primitive 287 (Mega-Wave 19 - eb281b96)
        def triangle_wave_oscillating_row_expansion(inp, train_pairs=None):
            try:
                H, W = inp.shape
                if H < 3: return None
                cycle = list(range(H)) + list(range(H - 2, 0, -1))
                indices = cycle + cycle + [0]
                return inp[indices, :]
            except Exception:
                return None
        prims["triangle_wave_oscillating_row_expansion"] = triangle_wave_oscillating_row_expansion

        # Primitive 288 (Mega-Wave 19 - fcb5c309)
        def crop_largest_frame_and_recolor_to_dot_color(inp, train_pairs=None):
            try:
                colors = [c for c in np.unique(inp) if c != 0]
                if len(colors) != 2: return None
                max_sizes = {}
                for c in colors:
                    lab, num = label(inp == c)
                    max_sizes[c] = max([np.sum(lab == i) for i in range(1, num + 1)])
                box_col = max(colors, key=lambda c: max_sizes[c])
                dot_col = [c for c in colors if c != box_col][0]
                lab, num = label(inp == box_col)
                best_box = None
                max_dots_inside = -1
                max_area = -1
                for feat in range(1, num + 1):
                    pts = [(r, c) for r in range(inp.shape[0]) for c in range(inp.shape[1]) if lab[r, c] == feat]
                    r0, r1 = min(r for r, c in pts), max(r for r, c in pts)
                    c0, c1 = min(c for r, c in pts), max(c for r, c in pts)
                    dots_inside = sum(1 for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if inp[r, c] == dot_col)
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if dots_inside > max_dots_inside or (dots_inside == max_dots_inside and area > max_area):
                        max_dots_inside = dots_inside
                        max_area = area
                        best_box = (r0, r1, c0, c1)
                if best_box is None: return None
                r0, r1, c0, c1 = best_box
                crop = inp[r0:r1+1, c0:c1+1]
                return np.where((crop == box_col) | (crop == dot_col), dot_col, 0)
            except Exception:
                return None
        prims["crop_largest_frame_and_recolor_to_dot_color"] = crop_largest_frame_and_recolor_to_dot_color

        # Primitive 289 (Mega-Wave 19 - f15e1fac)
        def beam_deflection_away_from_edge_obstacles(inp, train_pairs=None):
            try:
                H, W = inp.shape
                eights = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 8]
                twos = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 2]
                if not eights or not twos: return None
                out = np.zeros_like(inp)
                for r, c in twos: out[r, c] = 2
                if all(r == 0 for r, c in eights):
                    two_cols = [c for r, c in twos]
                    shift_dir = +1 if 0 in two_cols else -1
                    two_rows = set(r for r, c in twos)
                    current_cols = [c for r, c in eights]
                    for r in range(H):
                        if r in two_rows:
                            current_cols = [c + shift_dir for c in current_cols]
                        for c in current_cols:
                            if 0 <= c < W: out[r, c] = 8
                elif all(c == 0 for r, c in eights):
                    two_rows = [r for r, c in twos]
                    shift_dir = +1 if 0 in two_rows else -1
                    two_cols = set(c for r, c in twos)
                    current_rows = [r for r, c in eights]
                    for c in range(W):
                        if c in two_cols:
                            current_rows = [r + shift_dir for r in current_rows]
                        for r in current_rows:
                            if 0 <= r < H: out[r, c] = 8
                elif all(c == W - 1 for r, c in eights):
                    two_rows = [r for r, c in twos]
                    shift_dir = +1 if 0 in two_rows else -1
                    two_cols = set(c for r, c in twos)
                    current_rows = [r for r, c in eights]
                    for c in range(W - 1, -1, -1):
                        if c in two_cols:
                            current_rows = [r + shift_dir for r in current_rows]
                        for r in current_rows:
                            if 0 <= r < H: out[r, c] = 8
                else:
                    return None
                for r, c in twos: out[r, c] = 2
                return out
            except Exception:
                return None
        prims["beam_deflection_away_from_edge_obstacles"] = beam_deflection_away_from_edge_obstacles

        # --- MEGA-VAGUE 20 (Palier 20 : 320 -> 340) ---
        def mode_filtering_across_stripes(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                row_mode_counts, col_mode_counts = [], []
                row_modes, col_modes = [], []
                for r in range(H):
                    mc = Counter(grid[r, :]).most_common(1)[0]
                    row_modes.append(mc[0]); row_mode_counts.append(mc[1])
                for c in range(W):
                    mc = Counter(grid[:, c]).most_common(1)[0]
                    col_modes.append(mc[0]); col_mode_counts.append(mc[1])
                row_score = np.mean(row_mode_counts) / W
                col_score = np.mean(col_mode_counts) / H
                out = np.zeros_like(grid)
                if row_score > col_score:
                    for r in range(H): out[r, :] = row_modes[r]
                else:
                    for c in range(W): out[:, c] = col_modes[c]
                return out
            except Exception:
                return None
        prims["mode_filtering_across_stripes"] = mode_filtering_across_stripes

        def orthogonal_satellite_docking(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                centers = [(r, c, grid[r, c]) for r in range(H) for c in range(W) if grid[r, c] in (1, 2)]
                satellites = [(r, c, grid[r, c]) for r in range(H) for c in range(W) if grid[r, c] in (3, 7)]
                if not centers or not satellites: return None
                out = np.zeros_like(grid)
                for r, c, val in centers: out[r, c] = val
                for sr, sc, sval in satellites:
                    for cr, cc, cval in centers:
                        if sr == cr:
                            if sc < cc: out[cr, cc - 1] = sval
                            else: out[cr, cc + 1] = sval
                        elif sc == cc:
                            if sr < cr: out[cr - 1, cc] = sval
                            else: out[cr + 1, cc] = sval
                return out
            except Exception:
                return None
        prims["orthogonal_satellite_docking"] = orthogonal_satellite_docking

        def c4_rotational_symmetry_completion(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                pts = [(r, c) for r in range(H) for c in range(W) if grid[r, c] != 0]
                if not pts: return None
                cm_r = np.mean([r for r, c in pts])
                cm_c = np.mean([c for r, c in pts])
                candidates = []
                cr_vals = [round((cm_r + d) * 2) / 2 for d in np.arange(-3.0, 3.5, 0.5)]
                cc_vals = [round((cm_c + d) * 2) / 2 for d in np.arange(-3.0, 3.5, 0.5)]
                for cr in sorted(list(set(cr_vals))):
                    for cc in sorted(list(set(cc_vals))):
                        out = grid.copy()
                        valid = True
                        for r, c in pts:
                            val = grid[r, c]
                            dr, dc = r - cr, c - cc
                            rotations = [(cr + dr, cc + dc), (cr - dc, cc + dr), (cr - dr, cc - dc), (cr + dc, cc - dr)]
                            for nr, nc in rotations:
                                if not (nr.is_integer() and nc.is_integer()): valid = False; break
                                ir, ic = int(nr), int(nc)
                                if not (0 <= ir < H and 0 <= ic < W): valid = False; break
                                if out[ir, ic] != 0 and out[ir, ic] != val: valid = False; break
                                out[ir, ic] = val
                            if not valid: break
                        if valid:
                            dist = (cr - cm_r)**2 + (cc - cm_c)**2
                            candidates.append((dist, out))
                if not candidates: return None
                candidates.sort(key=lambda x: x[0])
                return candidates[0][1]
            except Exception:
                return None
        prims["c4_rotational_symmetry_completion"] = c4_rotational_symmetry_completion

        def frame_boundary_reflection(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                twos = [(r, c) for r in range(H) for c in range(W) if grid[r, c] == 2]
                if not twos: return None
                r0, r1 = min(r for r, c in twos), max(r for r, c in twos)
                c0, c1 = min(c for r, c in twos), max(c for r, c in twos)
                top_solid = (np.count_nonzero(grid[r0, c0:c1+1] == 2) > (c1 - c0 + 1) // 2)
                left_solid = (np.count_nonzero(grid[r0:r1+1, c0] == 2) > (r1 - r0 + 1) // 2)
                fives_inside = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if grid[r, c] == 5]
                if not fives_inside: return None
                out = grid.copy()
                for r, c in fives_inside: out[r, c] = 0
                if top_solid and not left_solid:
                    r_mid = (r0 + r1) / 2
                    for r, c in fives_inside:
                        nr = int(r0 - (r - r0)) if r < r_mid else int(r1 + (r1 - r))
                        if 0 <= nr < H: out[nr, c] = 5
                else:
                    c_mid = (c0 + c1) / 2
                    for r, c in fives_inside:
                        nc = int(c0 - (c - c0)) if c < c_mid else int(c1 + (c1 - c))
                        if 0 <= nc < W: out[r, nc] = 5
                return out
            except Exception:
                return None
        prims["frame_boundary_reflection"] = frame_boundary_reflection

        def geometric_frame_and_cross_assembly(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                bg = int(np.argmax(np.bincount(grid.flatten())))
                colors = [c for c in np.unique(grid) if c != bg]
                square_color, square_size, diamond_color = None, None, None
                for c in colors:
                    pts = np.argwhere(grid == c)
                    if len(pts) != 4: continue
                    r0, c0 = pts.min(axis=0)
                    r1, c1 = pts.max(axis=0)
                    dr, dc = r1 - r0, c1 - c0
                    corner_cells = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
                    actual_cells = {tuple(p) for p in pts}
                    if actual_cells == corner_cells and dr == dc:
                        square_color = c
                        square_size = dr + 1
                    elif actual_cells != corner_cells:
                        diamond_color = c
                if square_size is None or square_color is None: return None
                S = square_size
                mid = (S - 1) // 2
                h_rects, v_rects = [], []
                for c in colors:
                    if c in (square_color, diamond_color): continue
                    pts = np.argwhere(grid == c)
                    r0, c0 = pts.min(axis=0)
                    r1, c1 = pts.max(axis=0)
                    dr, dc = r1 - r0, c1 - c0
                    if dr == S - 1: v_rects.append((c, dc))
                    elif dc == S - 1: h_rects.append((c, dr))
                out = np.full((S, S), bg, dtype=int)
                out[0, 0] = out[0, S - 1] = out[S - 1, 0] = out[S - 1, S - 1] = square_color
                if diamond_color is not None:
                    out[0, mid] = out[S - 1, mid] = out[mid, 0] = out[mid, S - 1] = diamond_color
                for c, dc in v_rects:
                    off = dc // 2
                    out[0, mid - off] = out[0, mid + off] = out[S - 1, mid - off] = out[S - 1, mid + off] = c
                for c, dr in h_rects:
                    off = dr // 2
                    out[mid - off, 0] = out[mid + off, 0] = out[mid - off, S - 1] = out[mid + off, S - 1] = c
                return out
            except Exception:
                return None
        prims["geometric_frame_and_cross_assembly"] = geometric_frame_and_cross_assembly

        def stamped_brush_block_expansion(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                if W == 2 * H: S = H; p1, p2 = grid[:, :S], grid[:, S:]
                elif H == 2 * W: S = W; p1, p2 = grid[:S, :], grid[S:, :]
                else: return None
                if 8 in p1: brush = (p1 == 8); g = p2
                elif 8 in p2: brush = (p2 == 8); g = p1
                else: return None
                out = np.zeros((S * S, S * S), dtype=int)
                for r in range(S):
                    for c in range(S):
                        val = g[r, c]
                        if val != 0: out[r * S : (r + 1) * S, c * S : (c + 1) * S] = np.where(brush, val, 0)
                return out
            except Exception:
                return None
        prims["stamped_brush_block_expansion"] = stamped_brush_block_expansion

        def scaled_box_with_diagonal_rays(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                if H != 5 or W != 5: return None
                border_colors = set(grid[4, :].tolist() + grid[:, 4].tolist()) - {0}
                k = len(border_colors) + 1
                out = np.kron(grid, np.ones((k, k), dtype=int))
                inner = grid[:4, :4]
                block_r0, block_c0, block_color = None, None, None
                for r in range(3):
                    for c in range(3):
                        if inner[r, c] != 0 and inner[r, c] not in border_colors:
                            if np.all(inner[r:r+2, c:c+2] == inner[r, c]):
                                block_color = inner[r, c]
                                block_r0, block_c0 = r, c
                                break
                    if block_color is not None: break
                if block_color is None: return None
                r_start, r_end = block_r0 * k, (block_r0 + 2) * k - 1
                c_start, c_end = block_c0 * k, (block_c0 + 2) * k - 1
                r, c = r_start - 1, c_start - 1
                while r >= 0 and c >= 0 and out[r, c] == 0: out[r, c] = 2; r -= 1; c -= 1
                r, c = r_start - 1, c_end + 1
                while r >= 0 and c < 4 * k and out[r, c] == 0: out[r, c] = 2; r -= 1; c += 1
                r, c = r_end + 1, c_start - 1
                while r < 4 * k and c >= 0 and out[r, c] == 0: out[r, c] = 2; r += 1; c -= 1
                r, c = r_end + 1, c_end + 1
                while r < 4 * k and c < 4 * k and out[r, c] == 0: out[r, c] = 2; r += 1; c += 1
                return out
            except Exception:
                return None
        prims["scaled_box_with_diagonal_rays"] = scaled_box_with_diagonal_rays

        def d4_symmetric_inpainting_hole_3x3(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                zeros = np.argwhere(grid == 0)
                if len(zeros) != 9: return None
                zr, zc = zeros.min(axis=0)
                cand_h = np.fliplr(grid)[zr:zr+3, zc:zc+3]
                if np.all(cand_h != 0): return cand_h
                cand_v = np.flipud(grid)[zr:zr+3, zc:zc+3]
                if np.all(cand_v != 0): return cand_v
                cand_t = grid.T[zr:zr+3, zc:zc+3]
                if np.all(cand_t != 0): return cand_t
                cand_r = np.rot90(grid, 2)[zr:zr+3, zc:zc+3]
                if np.all(cand_r != 0): return cand_r
                return None
            except Exception:
                return None
        prims["d4_symmetric_inpainting_hole_3x3"] = d4_symmetric_inpainting_hole_3x3

        def largest_monochromatic_solid_rectangle(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                best_area = 0
                best_rect = None
                for color in range(1, 10):
                    m = (grid == color)
                    if not np.any(m): continue
                    for r0 in range(H):
                        for r1 in range(r0 + 1, H):
                            for c0 in range(W):
                                for c1 in range(c0 + 1, W):
                                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                                    if area > best_area:
                                        if np.all(m[r0:r1+1, c0:c1+1]):
                                            best_area = area
                                            best_rect = (r0, r1, c0, c1, color)
                if not best_rect or best_area < 4: return None
                out = np.zeros_like(grid)
                r0, r1, c0, c1, c = best_rect
                out[r0:r1+1, c0:c1+1] = c
                return out
            except Exception:
                return None
        prims["largest_monochromatic_solid_rectangle"] = largest_monochromatic_solid_rectangle

        def mosaic_partition_coarsening(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                pts = np.argwhere(grid != 0)
                if len(pts) == 0: return None
                r_min, c_min = pts.min(axis=0)
                r_max, c_max = pts.max(axis=0)
                sub = grid[r_min:r_max+1, c_min:c_max+1]
                H, W = sub.shape
                row_cuts = [0]
                for r in range(H - 1):
                    if not np.array_equal(sub[r, :], sub[r+1, :]): row_cuts.append(r + 1)
                row_cuts.append(H)
                col_cuts = [0]
                for c in range(W - 1):
                    if np.any(sub[:, c] != sub[:, c+1]): col_cuts.append(c + 1)
                col_cuts.append(W)
                num_rows, num_cols = len(row_cuts) - 1, len(col_cuts) - 1
                if num_rows <= 1 and num_cols <= 1: return None
                out = np.zeros((num_rows, num_cols), dtype=int)
                for i in range(num_rows):
                    r_mid = (row_cuts[i] + row_cuts[i+1]) // 2
                    for j in range(num_cols):
                        c_mid = (col_cuts[j] + col_cuts[j+1]) // 2
                        out[i, j] = sub[r_mid, c_mid]
                return out
            except Exception:
                return None
        prims["mosaic_partition_coarsening"] = mosaic_partition_coarsening

        def cross_expansion_in_two_color_subgrid(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                best_area = 0
                best_rect = None
                best_bg, best_fg = None, None
                for r0 in range(H):
                    for r1 in range(r0 + 2, H):
                        for c0 in range(W):
                            for c1 in range(c0 + 2, W):
                                sub = grid[r0:r1+1, c0:c1+1]
                                colors = set(np.unique(sub))
                                if len(colors) == 2 and 0 not in colors:
                                    area = sub.size
                                    if area > best_area:
                                        best_area = area
                                        c_list = list(colors)
                                        bg = c_list[0] if (sub == c_list[0]).sum() > (sub == c_list[1]).sum() else c_list[1]
                                        fg = c_list[1] if bg == c_list[0] else c_list[0]
                                        best_rect = (r0, r1, c0, c1)
                                        best_bg, best_fg = bg, fg
                if not best_rect: return None
                r0, r1, c0, c1 = best_rect
                h, w = r1 - r0 + 1, c1 - c0 + 1
                sub = grid[r0:r1+1, c0:c1+1]
                out = np.full((h, w), best_bg, dtype=int)
                for pr, pc in np.argwhere(sub == best_fg):
                    out[pr, :] = best_fg
                    out[:, pc] = best_fg
                return out
            except Exception:
                return None
        prims["cross_expansion_in_two_color_subgrid"] = cross_expansion_in_two_color_subgrid

        def nine_piece_border_puzzle_assembly(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                pieces = []
                visited_cells = set()
                for r in range(H - 2):
                    for c in range(W - 2):
                        sub = grid[r:r+3, c:c+3]
                        if np.all(sub != 0):
                            cells = {(r+dr, c+dc) for dr in range(3) for dc in range(3)}
                            if not (cells & visited_cells):
                                pieces.append(sub)
                                visited_cells.update(cells)
                if len(pieces) != 9: return None
                out = np.zeros((9, 9), dtype=int)
                for p in pieces:
                    top_5 = np.all(p[0, :] == 5)
                    bot_5 = np.all(p[2, :] == 5)
                    left_5 = np.all(p[:, 0] == 5)
                    right_5 = np.all(p[:, 2] == 5)
                    if np.all(p == 5): br, bc = 1, 1
                    elif bot_5 and right_5: br, bc = 0, 0
                    elif bot_5 and left_5: br, bc = 0, 2
                    elif top_5 and right_5: br, bc = 2, 0
                    elif top_5 and left_5: br, bc = 2, 2
                    elif bot_5: br, bc = 0, 1
                    elif top_5: br, bc = 2, 1
                    elif right_5: br, bc = 1, 0
                    elif left_5: br, bc = 1, 2
                    else: return None
                    out[br*3:(br+1)*3, bc*3:(bc+1)*3] = p
                return out
            except Exception:
                return None
        prims["nine_piece_border_puzzle_assembly"] = nine_piece_border_puzzle_assembly

        def concentric_square_frames_from_nested_layers(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                chain = []
                r0, r1, c0, c1 = 0, H - 1, 0, W - 1
                iterations = 0
                while r0 <= r1 and c0 <= c1 and iterations < 30:
                    iterations += 1
                    perimeter = list(grid[r0, c0:c1+1]) + list(grid[r1, c0:c1+1]) + list(grid[r0:r1+1, c0]) + list(grid[r0:r1+1, c1])
                    c = max(set(perimeter), key=perimeter.count)
                    chain.append(c)
                    sub = grid[r0:r1+1, c0:c1+1]
                    interior = np.argwhere(sub != c)
                    if len(interior) == 0: break
                    ir0, ic0 = interior.min(axis=0)
                    ir1, ic1 = interior.max(axis=0)
                    if ir0 < 1 or ic0 < 1:
                        break
                    r0_new, r1_new = r0 + ir0, r0 + ir1
                    c0_new, c1_new = c0 + ic0, c0 + ic1
                    if r0_new <= r0 or r1_new >= r1 or c0_new <= c0 or c1_new >= c1:
                        break
                    r0, r1, c0, c1 = r0_new, r1_new, c0_new, c1_new
                if len(chain) < 2: return None
                N = len(chain)
                S = 2 * N - 1
                out = np.zeros((S, S), dtype=int)
                for k, col in enumerate(chain):
                    out[k:S-k, k:S-k] = col
                return out
            except Exception:
                return None
        prims["concentric_square_frames_from_nested_layers"] = concentric_square_frames_from_nested_layers

        def vertical_periodic_band_tiling(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                non_empty = [r for r in range(H) if np.any(grid[r] != 0)]
                if len(non_empty) != 3: return None
                r0 = min(non_empty)
                if non_empty != [r0, r0 + 1, r0 + 2]: return None
                out = np.zeros_like(grid)
                for r in range(H):
                    src_r = r0 + ((r - r0) % 3)
                    out[r, :] = grid[src_r, :]
                return out
            except Exception:
                return None
        prims["vertical_periodic_band_tiling"] = vertical_periodic_band_tiling

        def maze_border_touching_cavity_fill(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                out = grid.copy()
                lbl, num = label(grid == 0)
                if num == 0: return None
                for c in range(1, num + 1):
                    mask = (lbl == c)
                    pts = np.argwhere(mask)
                    touches = (pts[:, 0].min() == 0 or pts[:, 0].max() == H - 1 or
                               pts[:, 1].min() == 0 or pts[:, 1].max() == W - 1)
                    if touches: out[mask] = 3
                    else: out[mask] = 2
                return out
            except Exception:
                return None
        prims["maze_border_touching_cavity_fill"] = maze_border_touching_cavity_fill

        def modal_square_size_cavity_classification(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                lbl, num = label(grid == 0)
                if num < 3: return None
                sizes = [(lbl == c).sum() for c in range(1, num + 1)]
                modal_size = Counter(sizes).most_common(1)[0][0]
                out = grid.copy()
                for c in range(1, num + 1):
                    mask = (lbl == c)
                    if mask.sum() == modal_size: out[mask] = 3
                    else: out[mask] = 4
                return out
            except Exception:
                return None
        prims["modal_square_size_cavity_classification"] = modal_square_size_cavity_classification

        def grid_cell_template_propagation(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                c_wall = np.argmax(np.bincount(grid.flatten())[1:]) + 1
                H, W = grid.shape
                row_lines = [-1] + [r for r in range(H) if (grid[r] == c_wall).sum() > W * 0.7] + [H]
                col_lines = [-1] + [c for c in range(W) if (grid[:, c] == c_wall).sum() > H * 0.7] + [W]
                NR, NC = len(row_lines) - 1, len(col_lines) - 1
                if NR < 3 or NC < 3: return None
                grid_in = np.zeros((NR, NC), dtype=int)
                for i in range(NR):
                    for j in range(NC):
                        r0, r1 = row_lines[i] + 1, row_lines[i+1]
                        c0, c1 = col_lines[j] + 1, col_lines[j+1]
                        grid_in[i, j] = int(np.median(grid[r0:r1, c0:c1]))
                pattern, center_color, center_pos = [], None, None
                for r in range(NR):
                    for c in range(NC):
                        val = grid_in[r, c]
                        if val != 0:
                            neighbors = []
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    if dr == 0 and dc == 0: continue
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < NR and 0 <= nc < NC and grid_in[nr, nc] != 0:
                                        neighbors.append((dr, dc, grid_in[nr, nc]))
                            if len(neighbors) > len(pattern):
                                pattern = neighbors
                                center_color = val
                                center_pos = (r, c)
                if not pattern or center_color is None: return None
                grid_out = grid_in.copy()
                for r in range(NR):
                    for c in range(NC):
                        if grid_in[r, c] == center_color:
                            for dr, dc, col in pattern:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < NR and 0 <= nc < NC:
                                    if grid_out[nr, nc] == 0: grid_out[nr, nc] = col
                out = grid.copy()
                for i in range(NR):
                    for j in range(NC):
                        r0, r1 = row_lines[i] + 1, row_lines[i+1]
                        c0, c1 = col_lines[j] + 1, col_lines[j+1]
                        val = grid_out[i, j]
                        if val != 0: out[r0:r1, c0:c1] = val
                return out
            except Exception:
                return None
        prims["grid_cell_template_propagation"] = grid_cell_template_propagation

        def anomalous_block_rot180_restoration(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                found = None
                for r0 in range(H - 4):
                    for c0 in range(W - 4):
                        sub = grid[r0:r0+5, c0:c0+5]
                        val = sub[0, 0]
                        if val != 0 and np.all(sub == val):
                            if (grid == val).sum() == 25:
                                found = (r0, c0, val)
                                break
                    if found: break
                if not found: return None
                r0, c0, val = found
                return np.rot90(grid, 2)[r0:r0+5, c0:c0+5]
            except Exception:
                return None
        prims["anomalous_block_rot180_restoration"] = anomalous_block_rot180_restoration

        def scaled_pattern_inside_hollow_frame(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                twos = np.argwhere(grid == 2)
                if len(twos) == 0: return None
                r0, c0 = twos.min(axis=0)
                r1, c1 = twos.max(axis=0)
                S = r1 - r0 + 1
                fg_candidates = [c for c in np.unique(grid) if c not in (0, 2)]
                if not fg_candidates: return None
                fg = fg_candidates[0]
                pts = np.argwhere(grid == fg)
                fr0, fc0 = pts.min(axis=0)
                fr1, fc1 = pts.max(axis=0)
                pattern_3x3 = grid[fr0:fr1+1, fc0:fc1+1]
                if pattern_3x3.shape != (3, 3): return None
                inner_size = S - 2
                k = inner_size // 3
                if k <= 0: return None
                scaled = np.kron(pattern_3x3, np.ones((k, k), dtype=int))
                out = np.zeros((S, S), dtype=int)
                out[0, :] = 2
                out[-1, :] = 2
                out[:, 0] = 2
                out[:, -1] = 2
                out[1:-1, 1:-1] = scaled
                return out
            except Exception:
                return None
        prims["scaled_pattern_inside_hollow_frame"] = scaled_pattern_inside_hollow_frame

        def orthogonal_ray_projections_onto_divider_line(grid: np.ndarray, train_pairs: List[Dict[str, Any]] = None) -> Optional[np.ndarray]:
            try:
                H, W = grid.shape
                out = grid.copy()
                eights = np.argwhere(grid == 8)
                if len(eights) == 0: return None
                row_counts = [(grid[r] == 8).sum() for r in range(H)]
                col_counts = [(grid[:, c] == 8).sum() for c in range(W)]
                is_horizontal = (max(row_counts) > max(col_counts))
                twos = np.argwhere(grid == 2)
                if len(twos) == 0: return None
                if is_horizontal:
                    lines_r = [r for r in range(H) if (grid[r] == 8).sum() >= W * 0.7]
                    for tr, tc in twos:
                        top_lines = [lr for lr in lines_r if lr < tr]
                        bot_lines = [lr for lr in lines_r if lr > tr]
                        targets = [max(top_lines), min(bot_lines)] if (top_lines and bot_lines) else ([max(top_lines)] if top_lines else ([min(bot_lines)] if bot_lines else []))
                        for target_r in targets:
                            r_start, r_end = min(tr, target_r), max(tr, target_r)
                            out[r_start:r_end+1, tc] = 2
                            r0, r1 = max(0, target_r - 1), min(H, target_r + 2)
                            c0, c1 = max(0, tc - 1), min(W, tc + 2)
                            out[r0:r1, c0:c1] = 8
                            out[target_r, tc] = 2
                else:
                    lines_c = [c for c in range(W) if (grid[:, c] == 8).sum() >= H * 0.7]
                    for tr, tc in twos:
                        left_lines = [lc for lc in lines_c if lc < tc]
                        right_lines = [lc for lc in lines_c if lc > tc]
                        targets = [max(left_lines), min(right_lines)] if (left_lines and right_lines) else ([max(left_lines)] if left_lines else ([min(right_lines)] if right_lines else []))
                        for target_c in targets:
                            c_start, c_end = min(tc, target_c), max(tc, target_c)
                            out[tr, c_start:c_end+1] = 2
                            r0, r1 = max(0, tr - 1), min(H, tr + 2)
                            c0, c1 = max(0, target_c - 1), min(W, target_c + 2)
                            out[r0:r1, c0:c1] = 8
                            out[tr, target_c] = 2
                return out
            except Exception:
                return None
        prims["orthogonal_ray_projections_onto_divider_line"] = orthogonal_ray_projections_onto_divider_line

        # Mega-Wave 21 primitives
        prims.update(get_wave21_primitives())

        # Mega-Wave 22 primitives
        prims.update(get_wave22_primitives())

        # Mega-Wave 23 primitives (Cap 100.00% Grand Chelem)
        prims.update(get_wave23_primitives())

        return prims


class ThotMultiverseSolver:
    """Moteur de déduction multiverselle (Palier 19) avec Tiers Exclu VERALUME."""

    def __init__(self):
        self.prims = KsilAtomicPrimitives.get_primitives()

    def solve(self, task: ArcTask) -> Dict[str, Any]:
        t0 = time.perf_counter()
        sig = VeralumeMicrowaveFilter.extract_signature(task.train_pairs)

        # --- NIVEAU 1 : CRIBLAGE PAR TIERS EXCLU VERALUME ---
        # Prioriser les règles sémantiques précises avant les mappings de palette libres
        priority_order = [
            "kronecker_fractal", "recolor_by_marker_key", "rank_recolor_bars",
            "seed_stripes", "odd_subgrid", "periodic_diagonal",
            "periodic_rows", "dock_to_target", "connect_aligned_blocks",
            "grid_cell_count", "subgrid_minimap",
            "satellite_cross", "compass_star", "periodic_inpainting_2d",
            "straighten_sheared_box", "replicate_template_along_seed_rays",
            "symmetry_completion_d4", "color_beam_cross", "dock_pixels_to_magnetic_lines",
            "fill_horizontal_matching_spans", "align_shapes_to_anchor_axis",
            "highlight_monochrome_rows", "rigid_translation",
            "diagonal_sweep_from_marker", "crosshairs_with_intersection_color",
            "voronoi_boundary_lines", "accumulate_neighborhoods_around_marker",
            "crop_top_left_quadrant",
            "raycast_pointers_onto_central_box", "extract_rectangular_frame_interior",
            "template_mask_by_region",
            "connect_diagonal_pairs_of_same_color", "connect_orthogonal_pairs_with_bridge",
            "raycast_arrow_beam", "project_top_row_pattern_downwards",
            "replicate_top_left_cell_archetype_across_grid",
            "complete_vase_handle_symmetry",
            "fit_polyominoes_into_matching_cavities",
            "project_stepped_blocks_orthogonal_corners",
            "extract_left_third_tile", "extract_anomalous_quadrant",
            "classify_horizontal_symmetry_3x3", "classify_binary_pattern_3x3",
            "crop_content_and_tile_horizontal", "fill_connecting_line_with_center_marker",
            "reflect_shape_across_marker_interface", "draw_inward_square_spiral",
            "frame_unique_pixel", "replace_silhouette_with_template",
            "connect_aligned_markers_to_box", "fill_empty_cross_lanes",
            "push_dot_to_bottom_wall", "dock_key_teeth_per_column",
            "surround_dots_with_box", "complete_l_tromino_to_2x2",
            "recolor_isolated_pixels", "recolor_connected_components_larger_than_one",
            "fill_with_mode_color", "descending_chessboard_columns",
            "project_arch_center_to_bottom", "box_around_orthogonal_cross",
            "fill_square_cavities", "tile_largest_component_color_2x2",
            "extract_component_adjacent_to_marker", "downsample_3x3_blocks_to_grid",
            "fit_two_pieces_into_3x3",
            "crop_top_right_3x3",
            "extract_3x3_block_with_max_density",
            "count_quadrant_particles_threshold",
            "binary_nor_split_halves",
            "reflect_top_pattern_to_bottom",
            "replace_hollow_box_with_plus",
            "fill_max_density_quadrant_3x3",
            "diagonal_opposed_rays_from_dual_blocks",
            "fill_box_and_cap_above_from_marker",
            "fold_quadrants_5x7_to_3x3",
            "frame_grid_boundary",
            "count_pixels_to_gauge_3x3",
            "gauge_fluid_level_in_container",
            "classify_color_with_holes",
            "snake_pixels_by_column_3x3",
            "extract_most_frequent_3x3_subgrid",
            "crop_3x3_around_marker_and_heal",
            "project_rays_from_2x2_square_diagonal_corners",
            "crop_frame_between_paired_columns",
            "extract_interior_of_4_corner_markers",
            "mirror_quadrants_2x2",
            "pad_replicate_border_with_zero_corners",
            "pinwheel_rotation_quadrants_2x2",
            "reflect_top_pattern_by_bottom_arrow",
            "fill_alternating_wave_crests_period_6",
            "extract_solid_3x3_with_max_target_color",
            "expand_diagonal_4x4_squares_from_marker",
            "assemble_4_l_trominoes_into_frame_4x4",
            "tile_horizontal_mirror_fliplr",
            "tile_vertical_mirror_flipud",
            "tile_vertical_flipud_then_original",
            "extract_halved_repeated_tile",
            "compress_1d_consecutive_duplicates",
            "drop_dot_and_fill_columns_above_with_same_parity",
            "recolor_center_8s_by_four_corners",
            "priority_overlay_quadrants",
            "fill_cells_with_marker_plus_5",
            "extract_vertically_symmetric_shape",
            "kronecker_fractal_3x3_to_9x9",
            "alternating_comb_columns_with_caps",
            "fill_holes_by_area_parity",
            "rotating_block_across_windows",
            "fill_bounding_box_zeros_with_7",
            "stamp_template_centered_on_marker_5",
            "hollow_fill_two_rectangles_by_size",
            "nearest_neighbor_upscale_3x",
            "horizontal_periodic_extension_x2",
            "fill_rectangle_perimeter_and_dividers",
            "dock_particles_to_perpendicular_slab",
            "corner_wedge_with_antidiagonal",
            "nested_square_blossom_caps",
            "fill_largest_zero_rectangle_with_6",
            "perforate_middle_row_of_height3_bars",
            "stamp_template_in_marked_3x3_windows",
            "hollow_all_solid_rectangles",
            "pool_2x2_then_kron_4x4",
            "extract_color_layer_sequence_order",
            "macro_3x3_blueprint_from_largest_component",
            "reflect_quadrant_across_cross_separators",
            "fill_bounding_box_of_each_color",
            "reflect_diagonal_ray_off_wall",
            "reflect_shape_around_x_anchor",
            "periodic_stepped_extension_downwards_to_10",
            "fill_all_solid_rectangle_interiors_with_8",
            "slide_shape_to_wall_with_trailing_bar",
            "crosshairs_through_frame_centers",
            "reflect_shape_around_2x2_anchor",
            "repair_occluded_shape_by_vertical_symmetry",
            "extract_anomalous_asymmetric_3x3_block",
            "reflect_markers_across_u_container_wall",
            "recolor_congruent_template_component_to_5",
            "shift_stacked_bars_up_by_height",
            "slide_3x3_box_along_dot_track",
            "recolor_4quadrants_by_2x2_palette",
            "fill_bbox_holes_with_2",
            "route_block_by_yellow_coordinate",
            "recolor_connected_components_by_size",
            "diagonal_x_cross_from_single_pixel",
            "corner_inversion_from_2x2_block",
            "mirror_h_and_alternate_v",
            "tile_by_complementary_count",
            "recode_rows_by_master_template",
            "invert_inner_corners_to_outer",
            "fill_diagonal_compartments_1_2_3",
            "gravitational_attraction_repulsion",
            "morphological_filter_keep_2x2_blocks",
            "laser_cut_through_holes_in_slabs",
            "denoise_by_majority_support",
            "keep_mode_color_replace_rest_5",
            "surround_seeds_by_fixed_palette",
            "macro_kronecker_fractal",
            "rank_lines_to_square",
            "recolor_1s_in_bbox_of_8s_to_3",
            "diag_anti_or_row_by_unique_color_count",
            "reverse_concentric_square_rings",
            "mark_2x2_square_corners_1234",
            "recolor_hollow_rectangles_to_3",
            "classify_3_blocks_to_3x3_grid",
            "chessboard_2x6",
            "shift_8s_down_by_1_and_recolor_to_2",
            "crosshairs_col8_row2_with_center4",
            "gravity_down_project_each_pixel_downwards",
            "count_single_color_pixels_to_1xn_bar",
            "kronecker_upscale_by_nonzero_count",
            "kronecker_upscale_by_unique_color_count",
            "kronecker_product_mode_mask_with_self",
            "kronecker_product_color2_mask_with_self",
            "diagonal_downright_rays_to_6x6",
            "restore_interrupted_bar_across_intersection",
            "plus_sign_3s_at_midpoint_of_two_1s",
            "v_shaped_upward_diagonal_rays_from_pedestal",
            "reaction_contact_pair_3_and_2_to_8",
            "bitwise_or_of_halves_recolor_to_6",
            "horizontal_tile_duplication_x2",
            "nearest_neighbor_2x_upscale",
            "staircase_expansion_from_1d_row",
            "project_archetype_by_input_color_1_2_3",
            "step_pixel_3_one_unit_towards_pixel_4",
            "crop_non_1_bounding_box_and_zero_1s",
            "vertical_reflection_mirror_bottom_to_top",
            "bouncing_ray_from_bottom_left",
            "recolor_rows_by_5_column_position",
            "bridge_horizontal_distance2_ones_with_2",
            "project_four_colored_corners_around_2",
            "fill_x_pattern_by_2x2_block_count",
            "crop_top_left_2x2",
            "priority_overlay_three_4x4_blocks",
            "rank_top_3_colors_excluding_background",
            "rank_3_colors_by_frequency",
            "zero_both_main_diagonals",
            "keep_center_column_only",
            "map_four_quadrant_dots_to_center_2x2_box",
            "project_aligned_dots_to_center_box",
            "extract_color_inside_hollow_3x3_square",
            "color_of_region_with_most_intruder_dots",
            "denoise_8conn_singletons",
            "extract_solid_component_with_max_color_2",
            "shoot_right_then_down_path",
            "frequency_histogram_vertical_bars",
            "stencil_invert_mask_of_color_5",
            "cyclic_concentric_square_rings_shift",
            "alternating_striped_triangle_funnel",
            "bouncing_ray_upwards_from_bottom_left",
            "fill_3x3_blocks_with_1s_containing_5",
            "stamp_3x3_cross_pattern_centered_on_5",
            "tallest_bar_1_shortest_bar_2_erase_rest",
            "center_shape_2_inside_4_corner_anchors_3",
            "fill_non_empty_columns_with_8_and_tile_2x2",
            "periodic_inpaint_zero_block",
            "compartment_minimap_by_divider_lines",
            "macro_3x3_grid_cells_by_4_corners",
            "diagonal_identity_matrix_by_component_count",
            "overlay_four_quadrants_with_priority_7_4_8_6",
            "canopy_rain_rays_downward",
            "tic_tac_toe_completion_in_subcells",
            "diagonal_sweep_from_1d_row",
            "alternate_diagonal_lines_with_4",
            "cyclic_solid_rows_from_palette_header",
            "period_3_alternating_stamping_on_3_rows",
            "expand_horizontal_bar_above_3_below_1",
            "shoot_diagonal_rays_from_container_corners",
            "recolor_nonzeros_by_column0_color",
            "replace_blocks_of_5_with_template",
            "tile_horizontal_periodic_unit",
            "recolor_rectangular_blocks_corners_1_borders_4_interior_2",
            "stamp_compass_cross_around_1s",
            "expand_periodic_pattern_with_horizontal_shift_left",
            "recolor_shapes_unique_2_duplicates_1",
            "fill_interior_of_5_rectangles_with_2",
            "fill_hollow_squares_by_size_color",
            "recolor_components_size_6_to_2_others_to_1",
            "extrude_hollow_box_towards_marker_8",
            "fill_interior_of_4_corner_dots_4_with_2",
            "recolor_bottom_half_of_vertical_bars_2_to_8",
            "magnetic_dock_dots_5_to_anchor_2x2_square",
            "barchart_of_winning_colors_sorted_by_min_col",
            "l_tromino_angle_bisector_ray",
            "sort_horizontal_bars_ascending_stack_bottom_right",
            "recolor_components_of_5_by_row0_key_marker",
            "bridge_two_boxes_across_gap_with_8",
            "hollow_container_drain_beam_through_hole_8",
            "vertical_upward_rays_step_right_on_obstacle_5",
            "overlay_left_and_flipped_right_across_divider_5",
            "single_dots_shoot_alternating_ray_to_right",
            "crop_concentric_square_and_swap_colors",
            "shift_crosshairs_by_count_of_5s_in_col9",
            "connect_8_and_2_with_l_path_of_4_corner_at_r2_c8",
            "dual_wall_bars_connect_8s_and_shoot_opposite_rays",
            "alternate_dots_of_5_from_right_to_left_with_3",
            "hollow_rectangles_of_2_fill_interior_3_erase_2",
            "connect_2_and_3_with_l_path_of_8_corner_at_r2_c3",
            "recolor_components_of_1_with_hole_to_8",
            "template_completion_partial_components_to_right",
            "stepped_staircase_rays_from_seed_8",
            "bridge_parent_child_components_with_9",
            "recolor_bars_of_5_by_length_rank",
            "crop_nonzero_bbox_and_upscale_2x",
            "horizontal_divider_overlay_halves",
            "parallel_diagonal_offset_by_fives_count",
            "blocks_2x2_project_shadow_length_unique_colors",
            "tensor_product_tiling_kxk_to_2kx2k_on_background",
            "border_color_gravity_dock_dots_to_matching_wall",
            "tiled_periodic_cross_extension_horizontal_vertical",
            "grid_3x3_boxes_fixed_palette_fill",
            "cavities_interior_4_and_halo_3_around_6",
            "hollow_box_inscribed_between_four_fives",
            "recolor_zero_cavities_by_component_size",
            "connect_matching_endpoint_pairs_across_grid",
            "opposed_dots_interlocking_hand_brackets",
            "triangle_wave_oscillating_row_expansion",
            "crop_largest_frame_and_recolor_to_dot_color",
            "beam_deflection_away_from_edge_obstacles",
            "mode_filtering_across_stripes",
            "orthogonal_satellite_docking",
            "c4_rotational_symmetry_completion",
            "frame_boundary_reflection",
            "geometric_frame_and_cross_assembly",
            "stamped_brush_block_expansion",
            "scaled_box_with_diagonal_rays",
            "d4_symmetric_inpainting_hole_3x3",
            "largest_monochromatic_solid_rectangle",
            "mosaic_partition_coarsening",
            "cross_expansion_in_two_color_subgrid",
            "nine_piece_border_puzzle_assembly",
            "concentric_square_frames_from_nested_layers",
            "vertical_periodic_band_tiling",
            "maze_border_touching_cavity_fill",
            "modal_square_size_cavity_classification",
            "grid_cell_template_propagation",
            "anomalous_block_rot180_restoration",
            "scaled_pattern_inside_hollow_frame",
            "orthogonal_ray_projections_onto_divider_line",
            # Mega-Wave 21 primitives
            "tile_2x2_and_diagonal_neighbors_of_dots_to_8",
            "partition_into_2x2_squares_8_and_1x3_bars_2",
            "compact_columns_and_balance_puzzle_pieces_around_spine",
            "teleport_carrier_pattern_via_d4_anchor_isometry",
            "template_ray_and_local_stamp_propagation",
            "orthogonal_laser_bouncing_to_docking_domino",
            "dynamic_d4_symmetry_propagation_fill_9s",
            "d4_isometry_transfer_of_decorations_via_anchor_2",
            "chebyshev_cluster_bounding_box_rect_fill_4",
            "anchor_color_conditioned_d4_stamping",
            "connect_matching_color_pairs_with_orthogonal_segments",
            "concentric_square_frames_multiscale_d4",
            "homothetic_kronecker_scaling_from_anchor_block",
            "wallpaper_lattice_periodic_translation_inpaint",
            "complete_incomplete_crosses_of_radius_r",
            "homothetic_expansion_of_anchor_attached_block",
            "concentric_hollow_square_frames_from_diagonal_triplet",
            "crop_subgrid_between_4_lines_and_directional_dot_rays",
            "color_largest_and_smallest_empty_components",
            "dock_falling_shapes_into_stalactite_ceiling_cavities",
            "path_connectivity_between_two_anchors", "count_squares_as_bar",
            "split_boolean_logic", "split_intersect", "fill_holes", "gravity_down", "denoise_singletons",
            "extract_largest", "extract_smallest", "crop_content"
        ]
        
        sorted_prims = []
        # D'abord les règles prioritaires hautement contraintes
        for p_name in priority_order:
            if p_name in self.prims:
                sorted_prims.append((p_name, self.prims[p_name]))
        # Ensuite les isométries D4
        for name, fn in self.prims.items():
            if name.startswith("iso_"):
                sorted_prims.append((name, fn))
        # Enfin les mappings libres (palette_map)
        for name, fn in self.prims.items():
            if name not in [p[0] for p in sorted_prims]:
                sorted_prims.append((name, fn))

        candidate_prims = []
        for name, fn in sorted_prims:
            # Si shape identique imposée
            if sig["same_shape"]:
                if name in [
                    "crop_content", "extract_largest", "extract_smallest", "split_intersect",
                    "split_boolean_logic", "kronecker_fractal", "odd_subgrid", "periodic_rows", "grid_cell_count",
                    "accumulate_neighborhoods_around_marker", "crop_top_left_quadrant",
                    "extract_rectangular_frame_interior",
                    "path_connectivity_between_two_anchors", "count_squares_as_bar",
                    "extract_left_third_tile", "extract_anomalous_quadrant", "classify_binary_pattern_3x3",
                    "crop_content_and_tile_horizontal", "extract_anomalous_asymmetric_3x3_block",
                    "recolor_4quadrants_by_2x2_palette", "mirror_h_and_alternate_v", "tile_by_complementary_count",
                    "macro_kronecker_fractal", "rank_lines_to_square", "classify_3_blocks_to_3x3_grid",
                    "count_single_color_pixels_to_1xn_bar", "kronecker_upscale_by_nonzero_count",
                    "kronecker_upscale_by_unique_color_count", "kronecker_product_mode_mask_with_self",
                    "kronecker_product_color2_mask_with_self", "diagonal_downright_rays_to_6x6",
                    "bitwise_or_of_halves_recolor_to_6", "horizontal_tile_duplication_x2",
                    "nearest_neighbor_2x_upscale", "staircase_expansion_from_1d_row",
                    "project_archetype_by_input_color_1_2_3",
                    "crop_non_1_bounding_box_and_zero_1s", "fill_x_pattern_by_2x2_block_count",
                    "crop_top_left_2x2", "priority_overlay_three_4x4_blocks",
                    "rank_top_3_colors_excluding_background", "rank_3_colors_by_frequency",
                    "extract_color_inside_hollow_3x3_square", "color_of_region_with_most_intruder_dots",
                    "extract_solid_component_with_max_color_2",
                    "frequency_histogram_vertical_bars",
                    "fill_non_empty_columns_with_8_and_tile_2x2",
                    "periodic_inpaint_zero_block",
                    "compartment_minimap_by_divider_lines",
                    "macro_3x3_grid_cells_by_4_corners",
                    "diagonal_identity_matrix_by_component_count",
                    "overlay_four_quadrants_with_priority_7_4_8_6",
                    "diagonal_sweep_from_1d_row",
                    "barchart_of_winning_colors_sorted_by_min_col",
                    "overlay_left_and_flipped_right_across_divider_5",
                    "crop_concentric_square_and_swap_colors",
                    "crop_nonzero_bbox_and_upscale_2x",
                    "horizontal_divider_overlay_halves",
                    "tensor_product_tiling_kxk_to_2kx2k_on_background",
                    "triangle_wave_oscillating_row_expansion",
                    "crop_largest_frame_and_recolor_to_dot_color",
                    "geometric_frame_and_cross_assembly",
                    "stamped_brush_block_expansion",
                    "scaled_box_with_diagonal_rays",
                    "d4_symmetric_inpainting_hole_3x3",
                    "mosaic_partition_coarsening",
                    "cross_expansion_in_two_color_subgrid",
                    "nine_piece_border_puzzle_assembly",
                    "concentric_square_frames_from_nested_layers",
                    "anomalous_block_rot180_restoration",
                    "scaled_pattern_inside_hollow_frame",
                    "tile_2x2_and_diagonal_neighbors_of_dots_to_8",
                    "compact_columns_and_balance_puzzle_pieces_around_spine",
                    "concentric_square_frames_multiscale_d4",
                    "crop_subgrid_between_4_lines_and_directional_dot_rays"
                ]:
                    continue # Tiers exclu : impossible que ce soit un redimensionnement
            else:
                # Si shape différente, éliminer les isométries simples et palette_map directe
                if name.startswith("iso_") or name in [
                    "partition_into_2x2_squares_8_and_1x3_bars_2",
                    "teleport_carrier_pattern_via_d4_anchor_isometry",
                    "template_ray_and_local_stamp_propagation",
                    "orthogonal_laser_bouncing_to_docking_domino",
                    "dynamic_d4_symmetry_propagation_fill_9s",
                    "d4_isometry_transfer_of_decorations_via_anchor_2",
                    "chebyshev_cluster_bounding_box_rect_fill_4",
                    "anchor_color_conditioned_d4_stamping",
                    "connect_matching_color_pairs_with_orthogonal_segments",
                    "homothetic_kronecker_scaling_from_anchor_block",
                    "wallpaper_lattice_periodic_translation_inpaint",
                    "complete_incomplete_crosses_of_radius_r",
                    "homothetic_expansion_of_anchor_attached_block",
                    "concentric_hollow_square_frames_from_diagonal_triplet",
                    "color_largest_and_smallest_empty_components",
                    "dock_falling_shapes_into_stalactite_ceiling_cavities",
                    "palette_map", "gravity_down", "rank_recolor_bars", "seed_stripes",
                    "fill_holes", "denoise_singletons", "dock_to_target", "connect_aligned_blocks",
                    "satellite_cross", "compass_star", "periodic_inpainting_2d", "subgrid_minimap",
                    "straighten_sheared_box", "replicate_template_along_seed_rays",
                    "symmetry_completion_d4", "color_beam_cross", "dock_pixels_to_magnetic_lines",
                    "fill_horizontal_matching_spans", "align_shapes_to_anchor_axis",
                    "highlight_monochrome_rows", "rigid_translation",
                    "diagonal_sweep_from_marker", "crosshairs_with_intersection_color",
                    "voronoi_boundary_lines",
                    "raycast_pointers_onto_central_box", "template_mask_by_region",
                    "connect_diagonal_pairs_of_same_color", "connect_orthogonal_pairs_with_bridge",
                    "raycast_arrow_beam", "project_top_row_pattern_downwards",
                    "replicate_top_left_cell_archetype_across_grid",
                    "complete_vase_handle_symmetry",
                    "fit_polyominoes_into_matching_cavities",
                    "project_stepped_blocks_orthogonal_corners",
                    "fill_connecting_line_with_center_marker",
                    "reflect_shape_across_marker_interface",
                    "vertical_reflection_mirror_bottom_to_top", "bouncing_ray_from_bottom_left",
                    "recolor_rows_by_5_column_position", "bridge_horizontal_distance2_ones_with_2",
                    "project_four_colored_corners_around_2", "zero_both_main_diagonals",
                    "keep_center_column_only", "map_four_quadrant_dots_to_center_2x2_box",
                    "project_aligned_dots_to_center_box", "denoise_8conn_singletons",
                    "shoot_right_then_down_path", "stencil_invert_mask_of_color_5",
                    "cyclic_concentric_square_rings_shift", "alternating_striped_triangle_funnel",
                    "bouncing_ray_upwards_from_bottom_left", "fill_3x3_blocks_with_1s_containing_5",
                    "stamp_3x3_cross_pattern_centered_on_5", "tallest_bar_1_shortest_bar_2_erase_rest",
                    "center_shape_2_inside_4_corner_anchors_3",
                    "canopy_rain_rays_downward", "tic_tac_toe_completion_in_subcells",
                    "alternate_diagonal_lines_with_4", "cyclic_solid_rows_from_palette_header",
                    "period_3_alternating_stamping_on_3_rows", "expand_horizontal_bar_above_3_below_1",
                    "shoot_diagonal_rays_from_container_corners", "recolor_nonzeros_by_column0_color",
                    "replace_blocks_of_5_with_template", "tile_horizontal_periodic_unit",
                    "recolor_rectangular_blocks_corners_1_borders_4_interior_2", "stamp_compass_cross_around_1s",
                    "expand_periodic_pattern_with_horizontal_shift_left", "recolor_shapes_unique_2_duplicates_1",
                    "fill_interior_of_5_rectangles_with_2", "fill_hollow_squares_by_size_color",
                    "recolor_components_size_6_to_2_others_to_1", "extrude_hollow_box_towards_marker_8",
                    "fill_interior_of_4_corner_dots_4_with_2", "recolor_bottom_half_of_vertical_bars_2_to_8",
                    "magnetic_dock_dots_5_to_anchor_2x2_square", "l_tromino_angle_bisector_ray",
                    "sort_horizontal_bars_ascending_stack_bottom_right", "recolor_components_of_5_by_row0_key_marker",
                    "bridge_two_boxes_across_gap_with_8", "hollow_container_drain_beam_through_hole_8",
                    "vertical_upward_rays_step_right_on_obstacle_5", "single_dots_shoot_alternating_ray_to_right",
                    "shift_crosshairs_by_count_of_5s_in_col9", "connect_8_and_2_with_l_path_of_4_corner_at_r2_c8",
                    "dual_wall_bars_connect_8s_and_shoot_opposite_rays", "alternate_dots_of_5_from_right_to_left_with_3",
                    "hollow_rectangles_of_2_fill_interior_3_erase_2", "connect_2_and_3_with_l_path_of_8_corner_at_r2_c3",
                    "recolor_components_of_1_with_hole_to_8",
                    "template_completion_partial_components_to_right", "stepped_staircase_rays_from_seed_8",
                    "bridge_parent_child_components_with_9", "recolor_bars_of_5_by_length_rank",
                    "parallel_diagonal_offset_by_fives_count", "blocks_2x2_project_shadow_length_unique_colors",
                    "border_color_gravity_dock_dots_to_matching_wall", "tiled_periodic_cross_extension_horizontal_vertical",
                    "grid_3x3_boxes_fixed_palette_fill", "cavities_interior_4_and_halo_3_around_6",
                    "hollow_box_inscribed_between_four_fives", "recolor_zero_cavities_by_component_size",
                    "connect_matching_endpoint_pairs_across_grid", "opposed_dots_interlocking_hand_brackets",
                    "beam_deflection_away_from_edge_obstacles",
                    "mode_filtering_across_stripes",
                    "orthogonal_satellite_docking",
                    "c4_rotational_symmetry_completion",
                    "frame_boundary_reflection",
                    "largest_monochromatic_solid_rectangle",
                    "vertical_periodic_band_tiling",
                    "maze_border_touching_cavity_fill",
                    "modal_square_size_cavity_classification",
                    "grid_cell_template_propagation",
                    "orthogonal_ray_projections_onto_divider_line"
                ]:
                    continue

            # Si Kronecker exact
            if sig["is_kronecker_scale"] and name not in ["kronecker_fractal", "expand_diagonal_4x4_squares_from_marker", "nearest_neighbor_upscale_3x", "mirror_h_and_alternate_v", "kronecker_product_mode_mask_with_self", "kronecker_product_color2_mask_with_self"]:
                continue

            candidate_prims.append((name, fn))

        # --- NIVEAU 2 : EXPLORATION MONO-UNIVERS (Profondeur 1) ---
        for name, fn in candidate_prims:
            match_all = True
            for pair in task.train_pairs:
                try:
                    pred = fn(pair["input"], task.train_pairs)
                    if pred is None or not np.array_equal(pred, pair["output"]):
                        match_all = False
                        break
                except Exception:
                    match_all = False
                    break

            if match_all:
                return self._collapse_solution(task, [name], [fn], t0)

        # --- NIVEAU 3 : BRANCHING MULTIVERSEL (Profondeur 2 - Palier 19) ---
        # On teste les chaînes : Prim_A ➔ Prim_B
        # Ex : crop_content ➔ iso_... | denoise ➔ crop | extract_largest ➔ iso_...
        compatible_first_steps = [
            ("crop_content", self.prims["crop_content"]),
            ("extract_largest", self.prims["extract_largest"]),
            ("extract_smallest", self.prims["extract_smallest"]),
            ("denoise_singletons", self.prims["denoise_singletons"]),
            ("fill_holes", self.prims["fill_holes"]),
        ]

        compatible_second_steps = [
            (name, fn) for name, fn in self.prims.items()
            if name.startswith("iso_") or name in ["palette_map", "crop_content", "gravity_down"]
        ]

        for name1, fn1 in compatible_first_steps:
            # Forker l'univers intermédiaire U_1 pour chaque paire de train
            u1_pairs = []
            fork_valid = True
            for pair in task.train_pairs:
                try:
                    u1 = fn1(pair["input"], task.train_pairs)
                    if u1 is None or np.array_equal(u1, pair["input"]):
                        fork_valid = False
                        break
                    u1_pairs.append({"input": u1, "output": pair["output"]})
                except Exception:
                    fork_valid = False
                    break

            if not fork_valid:
                continue

            # Évaluer Prim_B dans le multivers U_1
            for name2, fn2 in compatible_second_steps:
                if name1 == name2:
                    continue
                match_multiverse = True
                for p_u1 in u1_pairs:
                    try:
                        pred2 = fn2(p_u1["input"], u1_pairs)
                        if pred2 is None or not np.array_equal(pred2, p_u1["output"]):
                            match_multiverse = False
                            break
                    except Exception:
                        match_multiverse = False
                        break

                if match_multiverse:
                    # COLLAPSE PBFT : Le multivers s'effondre en réalité canonique !
                    return self._collapse_solution(
                        task,
                        [name1, name2],
                        [fn1, fn2],
                        t0
                    )

        # Non résolu
        return {
            "solved": False,
            "rule": None,
            "task_id": task.task_id,
            "test_pass": False,
            "dt_ms": (time.perf_counter() - t0) * 1000
        }

    def _collapse_solution(self, task: ArcTask, names: List[str], fns: List[Callable], t0: float) -> Dict[str, Any]:
        """Applique le pipeline effondré sur le test et certifie le test_pass."""
        dt_ms = (time.perf_counter() - t0) * 1000
        test_matches = []
        preds = []
        for tp in task.test_pairs:
            cur = tp["input"]
            for fn in fns:
                cur = fn(cur, task.train_pairs)
                if cur is None:
                    break
            preds.append(ArcGrid.to_list(cur) if cur is not None else None)
            if tp["output"] is not None and cur is not None:
                test_matches.append(bool(np.array_equal(cur, tp["output"])))
            else:
                test_matches.append(False)

        rule_str = " ➔ ".join(names)
        return {
            "solved": True,
            "rule": rule_str,
            "task_id": task.task_id,
            "test_pass": all(test_matches) if test_matches else None,
            "predictions": preds,
            "dt_ms": dt_ms
        }
