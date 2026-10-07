import sys, time, os
from thot_arc_core import ArcTask
from thot_ksil_multiverse import ThotMultiverseSolver
from palier_20_darwin import CANONICAL_BENCHMARK_TASKS

solver = ThotMultiverseSolver()
for idx, tid in enumerate(CANONICAL_BENCHMARK_TASKS):
    print(f"[{idx+1}/360] {tid} starting...", flush=True)
    task = ArcTask.load_from_file(f"training/{tid}.json")
    t0 = time.time()
    res = solver.solve(task)
    dt = time.time() - t0
    print(f"[{idx+1}/360] {tid}: solved={res['solved']} in {dt:.3f}s", flush=True)
    if not res['solved']:
        print(f"--> FAILED on {tid}!", flush=True)
        break
