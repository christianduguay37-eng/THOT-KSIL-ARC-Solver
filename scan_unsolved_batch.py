import sys
import os
import glob
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from thot_arc_core import ArcTask
from palier_20_darwin import CANONICAL_BENCHMARK_TASKS
from thot_ksil_multiverse import ThotMultiverseSolver

solver = ThotMultiverseSolver()
train_dir = "Atelier_ARC_Prize/training"
files = sorted(glob.glob(os.path.join(train_dir, "*.json")))

unsolved = []
for f in files:
    task = ArcTask.load_from_file(f)
    if task.task_id in CANONICAL_BENCHMARK_TASKS:
        continue
    res = solver.solve(task)
    if not (res["solved"] and res["test_pass"]):
        unsolved.append(task)

print(f"Total tâches restantes à conquérir : {len(unsolved)}")

# Analyser les types de signatures sur les 10 premières
for i, task in enumerate(unsolved[:10]):
    in_s = task.train_pairs[0]["input"].shape
    out_s = task.train_pairs[0]["output"].shape
    in_c = set(task.train_pairs[0]["input"].flatten())
    out_c = set(task.train_pairs[0]["output"].flatten())
    same_shape = all(p["input"].shape == p["output"].shape for p in task.train_pairs)
    print(f"[{i+1:02d}] Tâche {task.task_id} | SameShape: {same_shape} | Shape: {in_s} -> {out_s} | InColors: {in_c} | OutColors: {out_c}")
