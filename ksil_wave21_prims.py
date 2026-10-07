import numpy as np
from typing import Dict, Callable, List
from collections import Counter
from scipy.ndimage import label
import itertools

from test_10fcaaa3 import solve_10fcaaa3
from test_150deff5 import solve_150deff5
from test_234bbc79 import solve_234bbc79
from test_0e206a2e import solve_0e206a2e
from test_264363fd import solve_264363fd
from test_2dd70a9a import solve_2dd70a9a
from test_3631a71a import solve_3631a71a
from test_36d67576 import solve_36d67576
from test_36fdfd69 import solve_36fdfd69
from test_3e980e27 import solve_3e980e27
from test_40853293 import solve_40853293
from test_4290ef0e import solve_4290ef0e
from test_447fd412 import solve_447fd412
from test_484b58aa import solve_484b58aa
from test_50846271 import solve_50846271
from test_57aa92db import solve_57aa92db
from test_5c2c9af4 import solve_5c2c9af4
from test_5daaa586 import solve_5daaa586
from test_6455b5f5 import solve_6455b5f5
from test_6a1e5592 import solve_6a1e5592

def wrap(solver):
    def wrapper(grid: np.ndarray, train_pairs=None):
        try:
            return solver(np.array(grid))
        except Exception:
            return None
    return wrapper

def get_wave21_primitives() -> Dict[str, Callable]:
    return {
        "tile_2x2_and_diagonal_neighbors_of_dots_to_8": wrap(solve_10fcaaa3),
        "partition_into_2x2_squares_8_and_1x3_bars_2": wrap(solve_150deff5),
        "compact_columns_and_balance_puzzle_pieces_around_spine": wrap(solve_234bbc79),
        "teleport_carrier_pattern_via_d4_anchor_isometry": wrap(solve_0e206a2e),
        "template_ray_and_local_stamp_propagation": wrap(solve_264363fd),
        "orthogonal_laser_bouncing_to_docking_domino": wrap(solve_2dd70a9a),
        "dynamic_d4_symmetry_propagation_fill_9s": wrap(solve_3631a71a),
        "d4_isometry_transfer_of_decorations_via_anchor_2": wrap(solve_36d67576),
        "chebyshev_cluster_bounding_box_rect_fill_4": wrap(solve_36fdfd69),
        "anchor_color_conditioned_d4_stamping": wrap(solve_3e980e27),
        "connect_matching_color_pairs_with_orthogonal_segments": wrap(solve_40853293),
        "concentric_square_frames_multiscale_d4": wrap(solve_4290ef0e),
        "homothetic_kronecker_scaling_from_anchor_block": wrap(solve_447fd412),
        "wallpaper_lattice_periodic_translation_inpaint": wrap(solve_484b58aa),
        "complete_incomplete_crosses_of_radius_r": wrap(solve_50846271),
        "homothetic_expansion_of_anchor_attached_block": wrap(solve_57aa92db),
        "concentric_hollow_square_frames_from_diagonal_triplet": wrap(solve_5c2c9af4),
        "crop_subgrid_between_4_lines_and_directional_dot_rays": wrap(solve_5daaa586),
        "color_largest_and_smallest_empty_components": wrap(solve_6455b5f5),
        "dock_falling_shapes_into_stalactite_ceiling_cavities": wrap(solve_6a1e5592),
    }
