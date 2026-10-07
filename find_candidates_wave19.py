import os, glob, json
from palier_20_darwin import CANONICAL_BENCHMARK_TASKS
from thot_arc_core import ArcTask
from thot_ksil_multiverse import ThotMultiverseSolver

solver = ThotMultiverseSolver()
train_dir = os.path.join(os.path.dirname(__file__), "training")
files = sorted(glob.glob(os.path.join(train_dir, "*.json")))

canonical_set = set(CANONICAL_BENCHMARK_TASKS)
print(f"Total canonical: {len(canonical_set)}")

unsolved = []
for f in files:
    tid = os.path.basename(f).replace(".json", "")
    if tid in canonical_set:
        continue
    task = ArcTask.load_from_file(f)
    res = solver.solve(task)
    if not (res["solved"] and res["test_pass"]):
        unsolved.append(task)

print(f"Total unsolved: {len(unsolved)}")

# Categorize unsolved by shape relation: same_shape vs diff_shape
same_shape = []
diff_shape = []
for t in unsolved:
    is_same = all(p["input"].shape == p["output"].shape for p in t.train_pairs)
    if is_same:
        same_shape.append(t)
    else:
        diff_shape.append(t)

print(f"Unsolved same_shape: {len(same_shape)}, diff_shape: {len(diff_shape)}")
print("\n--- Next 15 same_shape candidates ---")
for t in same_shape[:15]:
    inp_s = t.train_pairs[0]["input"].shape
    in_cols = sorted(list(set(t.train_pairs[0]["input"].flatten())))
    out_cols = sorted(list(set(t.train_pairs[0]["output"].flatten())))
    print(f"ID: {t.task_id} | Shape: {inp_s} | InCols: {in_cols} -> OutCols: {out_cols} | Trains: {len(t.train_pairs)}")

print("\n--- Next 15 diff_shape candidates ---")
for t in diff_shape[:15]:
    inp_s = t.train_pairs[0]["input"].shape
    out_s = t.train_pairs[0]["output"].shape
    in_cols = sorted(list(set(t.train_pairs[0]["input"].flatten())))
    out_cols = sorted(list(set(t.train_pairs[0]["output"].flatten())))
    print(f"ID: {t.task_id} | InShape: {inp_s} -> OutShape: {out_s} | InCols: {in_cols} -> OutCols: {out_cols} | Trains: {len(t.train_pairs)}")
