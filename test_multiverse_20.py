import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from thot_arc_core import ArcTask
from thot_ksil_multiverse import ThotMultiverseSolver

solver = ThotMultiverseSolver()
tids = [
    "90f3ed37", "d06dbe63", "ef135b50", "ea32f347", "f25fbde4",
    "e98196ab", "a78176bb", "fcc82909", "539a4f51", "d687bc17",
    "e21d9049", "272f95fa", "543a7ed5", "928ad970", "e8593010",
    "6cdd2623", "b7249182", "eb281b96", "fcb5c309", "f15e1fac"
]

passed = 0
for tid in tids:
    task = ArcTask.load_from_file(f"training/{tid}.json")
    res = solver.solve(task)
    if res["solved"] and res["test_pass"]:
        passed += 1
        print(f"✅ {tid}: {res['rule']} ({res['dt_ms']:.2f} ms)")
    else:
        print(f"❌ {tid}: solved={res['solved']}, test_pass={res['test_pass']}")

print(f"\nTOTAL MULTIVERSE PASSED: {passed}/{len(tids)}")
