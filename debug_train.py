import sys
from thot_arc_core import ArcTask, ArcGrid
import numpy as np

task = ArcTask.load_from_file("Atelier_ARC_Prize/training/0a938d79.json")
for i, p in enumerate(task.train_pairs):
    coords = np.argwhere(p["input"] != 0)
    print(f"Train {i+1} coords count:", len(coords), "colors:", [p["input"][r, c] for r, c in coords])

task2 = ArcTask.load_from_file("Atelier_ARC_Prize/training/0b148d64.json")
for i, p in enumerate(task2.train_pairs):
    objs = ArcGrid.get_connected_components(p["input"], background=0)
    print(f"Task 0b148d64 Train {i+1} objs count:", len(objs), "colors:", [o["color"] for o in objs], "areas:", [o["area"] for o in objs])
