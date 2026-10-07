import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from thot_arc_core import ArcTask

def inspect_details(tids):
    for tid in tids:
        fpath = f"Atelier_ARC_Prize/training/{tid}.json"
        task = ArcTask.load_from_file(fpath)
        print(f"\n{'='*55}")
        print(f"🎯 FORENSIC INSPECTION : {tid}")
        print(f"{'='*55}")
        for i, p in enumerate(task.train_pairs[:2]):
            print(f"\n--- Train {i+1} ---")
            print(f"IN ({p['input'].shape}):")
            for r in p["input"]: print(" ".join(f"{x:1d}" for x in r))
            print(f"OUT ({p['output'].shape}):")
            for r in p["output"]: print(" ".join(f"{x:1d}" for x in r))

inspect_details(["09629e4f", "0e206a2e"])
