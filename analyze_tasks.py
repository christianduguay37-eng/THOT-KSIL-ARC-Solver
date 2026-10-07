import os
import sys
import glob

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from thot_arc_core import ArcTask

files = sorted(glob.glob('Atelier_ARC_Prize/training/*.json'))[:15]
for idx, f in enumerate(files):
    task = ArcTask.load_from_file(f)
    p0 = task.train_pairs[0]
    in_s = p0["input"].shape
    out_s = p0["output"].shape
    in_c = set(p0["input"].flatten())
    out_c = set(p0["output"].flatten())
    print(f"[{idx+1:02d}] Tâche {task.task_id} : Shape {in_s} -> {out_s} | Colors {in_c} -> {out_c}")
