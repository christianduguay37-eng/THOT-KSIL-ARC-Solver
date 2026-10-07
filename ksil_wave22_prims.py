import numpy as np
from typing import Dict, Callable

from test_6aa20dc0 import solve_6aa20dc0
from test_6cf79266 import solve_6cf79266
from test_6ecd11f4 import solve_6ecd11f4
from test_72322fa7 import solve_72322fa7
from test_73251a56 import solve_73251a56
from test_776ffc46 import solve_776ffc46
from test_7df24a62 import solve_7df24a62
from test_846bdb03 import solve_846bdb03
from test_890034e9 import solve_890034e9
from test_8a004b2b import solve_8a004b2b
from test_97a05b5b import solve_97a05b5b
from test_98cf29f8 import solve_98cf29f8
from test_9aec4887 import solve_9aec4887
from test_9d9215db import solve_9d9215db
from test_9edfc990 import solve_9edfc990
from test_9f236235 import solve_9f236235
from test_a64e4611 import solve_a64e4611
from test_a8d7556c import solve_a8d7556c
from test_aba27056 import solve_aba27056
from test_b27ca6d3 import solve_b27ca6d3

def wrap(solver):
    def wrapper(grid: np.ndarray, train_pairs=None):
        try:
            return solver(np.array(grid))
        except Exception:
            return None
    return wrapper

def get_wave22_primitives() -> Dict[str, Callable]:
    return {
        "multiscale_d4_template_filler_between_anchor_pairs": wrap(solve_6aa20dc0),
        "non_overlapping_empty_square_cavity_filling_color1": wrap(solve_6cf79266),
        "mask_palette_by_macroblock_grid_occupancy": wrap(solve_6ecd11f4),
        "bidirectional_completion_dual_center_satellite_templates": wrap(solve_72322fa7),
        "symmetric_matrix_inpainting_via_diagonal_reflection": wrap(solve_73251a56),
        "recolor_objects_matching_template_inside_closed_frame": wrap(solve_776ffc46),
        "enclose_d4_isomorphic_constellations_with_padded_boxes": wrap(solve_7df24a62),
        "embed_bicolored_object_inside_box_with_auto_hflip": wrap(solve_846bdb03),
        "replicate_hollow_frame_around_empty_cavity_hole": wrap(solve_890034e9),
        "assemble_upscaled_subgrid_matching_anchor_blocks": wrap(solve_8a004b2b),
        "key_in_lock_jigsaw_insert_decorated_pieces_into_cavities": wrap(solve_97a05b5b),
        "retractable_winch_glide_mobile_block_along_stem_to_contact": wrap(solve_98cf29f8),
        "inscribe_8_pattern_into_multicolored_frame_with_edge_walls": wrap(solve_9aec4887),
        "concentric_d4_symmetric_square_frames_corners_and_edges": wrap(solve_9d9215db),
        "flood_fill_color_1_into_connected_empty_space": wrap(solve_9edfc990),
        "macro_grid_lattice_horizontal_reflection": wrap(solve_9f236235),
        "corridor_network_mapping_in_empty_space_between_clusters": wrap(solve_a64e4611),
        "maximal_nonoverlapping_empty_rectangle_cavity_filling_color2": wrap(solve_a8d7556c),
        "light_beam_cone_emanating_from_open_cavity_aperture": wrap(solve_aba27056),
        "frame_connected_dominoes_of_color2_with_color3_border": wrap(solve_b27ca6d3),
    }
