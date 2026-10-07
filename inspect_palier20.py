import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from thot_arc_core import ArcTask

def inspect_unsolved(task_ids):
    for tid in task_ids:
        fpath = os.path.join("Atelier_ARC_Prize", "training", f"{tid}.json")
        if not os.path.exists(fpath):
            continue
        task = ArcTask.load_from_file(fpath)
        print(f"\n{'='*50}")
        print(f"🔍 INSPECTION DE L'ÉNIGME : {tid}")
        print(f"{'='*50}")
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

inspect_unsolved(["017c7c7b", "025d127b", "045e512c", "0a938d79", "0b148d64"])
