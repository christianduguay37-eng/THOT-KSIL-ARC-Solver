import os, glob, json
import numpy as np
from palier_20_darwin import CANONICAL_BENCHMARK_TASKS
from thot_arc_core import ArcTask
from thot_ksil_multiverse import ThotMultiverseSolver

canonical_set = set(CANONICAL_BENCHMARK_TASKS)
# Also add the 15 we already have
current_15 = {
    "90f3ed37", "d06dbe63", "ef135b50", "ea32f347", "f25fbde4",
    "e98196ab", "a78176bb", "fcc82909", "539a4f51", "d687bc17",
    "e21d9049", "272f95fa", "543a7ed5", "928ad970", "e8593010"
}

files = sorted(glob.glob("training/*.json"))
unsolved = []
for f in files:
    tid = os.path.basename(f).replace(".json", "")
    if tid in canonical_set or tid in current_15:
        continue
    with open(f) as fp:
        d = json.load(fp)
    unsolved.append((tid, d))

print(f"Remaining unsolved: {len(unsolved)}")
for tid, d in unsolved[:25]:
    tr0_in = np.array(d["train"][0]["input"])
    tr0_out = np.array(d["train"][0]["output"])
    n_train = len(d["train"])
    in_cols = sorted(list(np.unique(tr0_in)))
    out_cols = sorted(list(np.unique(tr0_out)))
    print(f"ID: {tid} | Trains: {n_train} | Shapes: {tr0_in.shape} -> {tr0_out.shape} | Cols: {in_cols} -> {out_cols}")
