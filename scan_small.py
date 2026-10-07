import os, glob, json
import numpy as np
from palier_20_darwin import CANONICAL_BENCHMARK_TASKS

canonical_set = set(CANONICAL_BENCHMARK_TASKS)
current_16 = {
    "90f3ed37", "d06dbe63", "ef135b50", "ea32f347", "f25fbde4",
    "e98196ab", "a78176bb", "fcc82909", "539a4f51", "d687bc17",
    "e21d9049", "272f95fa", "543a7ed5", "928ad970", "e8593010",
    "6cdd2623"
}

files = sorted(glob.glob("training/*.json"))
candidates = []
for f in files:
    tid = os.path.basename(f).replace(".json", "")
    if tid in canonical_set or tid in current_16:
        continue
    with open(f) as fp:
        d = json.load(fp)
    
    # Check simple characteristics
    # small grids (<= 15x15)
    tr_shapes = [np.array(p["input"]).shape for p in d["train"]]
    out_shapes = [np.array(p["output"]).shape for p in d["train"]]
    max_h = max(s[0] for s in tr_shapes)
    max_w = max(s[1] for s in tr_shapes)
    
    candidates.append((tid, len(d["train"]), tr_shapes[0], out_shapes[0], max_h, max_w, d))

# Sort candidates: smaller grids first
candidates.sort(key=lambda x: (x[4] * x[5]))

print(f"Total available: {len(candidates)}")
print("\nSmallest grid tasks:")
for tid, n_tr, in_s, out_s, mh, mw, d in candidates[:25]:
    in_cols = sorted(list(np.unique(d["train"][0]["input"])))
    out_cols = sorted(list(np.unique(d["train"][0]["output"])))
    print(f"ID: {tid} | Trains: {n_tr} | MaxDim: {mh}x{mw} | {in_s}->{out_s} | Cols: {in_cols}->{out_cols}")
