import time, sys
from thot_arc_core import ArcTask
from thot_ksil_multiverse import ThotMultiverseSolver
from palier_20_darwin import CANONICAL_BENCHMARK_TASKS

solver = ThotMultiverseSolver()
print(f"Total benchmark tasks: {len(CANONICAL_BENCHMARK_TASKS)}")
for idx in range(340, len(CANONICAL_BENCHMARK_TASKS)):
    tid = CANONICAL_BENCHMARK_TASKS[idx]
    t0 = time.time()
    task = ArcTask.load_from_file(f"training/{tid}.json")
    res = solver.solve(task)
    dt = time.time() - t0
    print(f"[{idx+1}/360] {tid}: solved={res['solved']} in {dt:.3f}s", flush=True)
