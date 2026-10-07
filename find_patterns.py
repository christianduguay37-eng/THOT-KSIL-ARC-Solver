import glob, json
import numpy as np
from thot_arc_core import ArcTask
from thot_ksil_multiverse import ThotMultiverseSolver

solver = ThotMultiverseSolver()
all_files = sorted(glob.glob('training/*.json'))
unsolved = []
for f in all_files:
    task = ArcTask.load_from_file(f)
    res = solver.solve(task)
    if not (res['solved'] and res['test_pass']):
        unsolved.append(task)

print(f"Total unsolved: {len(unsolved)}")

for t in unsolved:
    # 1. Exact subwindow
    is_subwindow = True
    for p in t.train_pairs:
        inp = p['input']
        out = p['output']
        H_i, W_i = inp.shape
        H_o, W_o = out.shape
        found = False
        if H_o <= H_i and W_o <= W_i:
            for r in range(H_i - H_o + 1):
                for c in range(W_i - W_o + 1):
                    if np.array_equal(inp[r:r+H_o, c:c+W_o], out):
                        found = True
                        break
                if found:
                    break
        if not found:
            is_subwindow = False
            break
    if is_subwindow:
        print(f"SUBWINDOW: {t.task_id} In {t.train_pairs[0]['input'].shape} -> Out {t.train_pairs[0]['output'].shape}")

    # 2. Output is 1x1 color
    out_shapes = [p['output'].shape for p in t.train_pairs]
    if all(s == (1, 1) for s in out_shapes):
        print(f"OUTPUT_1x1: {t.task_id}")

    # 3. Small diff count (few pixels changed between input and output when same shape)
    same_shape = all(p['input'].shape == p['output'].shape for p in t.train_pairs)
    if same_shape:
        diff_counts = [np.sum(p['input'] != p['output']) for p in t.train_pairs]
        if max(diff_counts) <= 15:
            print(f"FEW_PIXELS_DIFF: {t.task_id} max_diff={max(diff_counts)} In {t.train_pairs[0]['input'].shape}")
