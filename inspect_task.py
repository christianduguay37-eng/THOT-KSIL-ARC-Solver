import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from thot_arc_core import ArcTask

def inspect_task(task_id: str):
    fpath = os.path.join("Atelier_ARC_Prize", "training", f"{task_id}.json")
    task = ArcTask.load_from_file(fpath)
    print(f"\n==========================================")
    print(f"🔍 INSPECTION DE LA TÂCHE {task_id}")
    print(f"==========================================")
    for i, p in enumerate(task.train_pairs[:2]):
        print(f"\n--- Train {i+1} ---")
        inp = p["input"]
        out = p["output"]
        print(f"INPUT ({inp.shape[0]}x{inp.shape[1]}):")
        for r in inp:
            print(" ".join(f"{x:1d}" for x in r))
        print(f"OUTPUT ({out.shape[0]}x{out.shape[1]}):")
        for r in out:
            print(" ".join(f"{x:1d}" for x in r))

inspect_task("05269061")
inspect_task("08ed6ac7")
