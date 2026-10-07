import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from thot_arc_core import ArcTask

def display_task(task_id: str):
    fpath = os.path.join("Atelier_ARC_Prize", "training", f"{task_id}.json")
    task = ArcTask.load_from_file(fpath)
    print(f"\n{'='*55}")
    print(f"🎯 ANALYSE FORENSIQUE : TÂCHE {task_id}")
    print(f"{'='*55}")
    for i, p in enumerate(task.train_pairs):
        inp = p["input"]
        out = p["output"]
        print(f"\n--- Exemple Train {i+1} ---")
        print(f"INPUT ({inp.shape[0]}x{inp.shape[1]}):")
        for r in inp:
            print(" ".join(f"{x:1d}" for x in r))
        print(f"OUTPUT ({out.shape[0]}x{out.shape[1]}):")
        for r in out:
            print(" ".join(f"{x:1d}" for x in r))
            
    print(f"\n--- TEST ---")
    for i, tp in enumerate(task.test_pairs):
        inp = tp["input"]
        out = tp["output"]
        print(f"TEST INPUT ({inp.shape[0]}x{inp.shape[1]}):")
        for r in inp:
            print(" ".join(f"{x:1d}" for x in r))
        if out is not None:
            print(f"TEST OUTPUT ({out.shape[0]}x{out.shape[1]}):")
            for r in out:
                print(" ".join(f"{x:1d}" for x in r))

display_task("017c7c7b")
