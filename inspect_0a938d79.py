from thot_arc_core import ArcTask
import numpy as np

task = ArcTask.load_from_file("Atelier_ARC_Prize/training/0a938d79.json")
for i, p in enumerate(task.train_pairs):
    coords = np.argwhere(p["input"] != 0)
    coords = sorted(coords, key=lambda pt: pt[1])
    c1, c2 = coords[0][1], coords[1][1]
    col1, col2 = p["input"][coords[0][0], c1], p["input"][coords[1][0], c2]
    # Vérifier où sont les colonnes dans output
    out_cols = [c for c in range(p["output"].shape[1]) if np.any(p["output"][:, c] != 0)]
    print(f"Train {i+1}: in cols ({c1}, {c2}) colors ({col1}, {col2}) | out cols:", out_cols)
