import time
import sys
from thot_arc_core import ArcTask
from thot_ksil_multiverse import ThotMultiverseSolver

def test_wave20():
    solver = ThotMultiverseSolver()
    wave20_tasks = [
        "e26a3af2", "ae3edfdc", "e40b9e2f", "f8a8fe49", "c8cbb738", "b190f7f5", "469497ad", "9ecd008a", "91714a58", "90c28cc7",
        "8731374e", "a8c38be5", "eb5a1d5d", "8eb1be9a", "7b6016b9", "83302e8f", "39e1d7f9", "ff805c23", "6b9890af", "ecdecbb3"
    ]

    print("=== TEST INTÉGRATION WAVE 20 DANS THOT MULTIVERSE SOLVER ===")
    all_ok = True
    for tid in wave20_tasks:
        t0 = time.perf_counter()
        task = ArcTask.load_from_file(f"training/{tid}.json")
        res = solver.solve(task)
        dt = (time.perf_counter() - t0) * 1000
        status = "PASS" if res.get("solved") and res.get("test_pass") else "FAIL"
        if status != "PASS":
            all_ok = False
        print(f"[{status}] {tid} | solved={res.get('solved')} | test_pass={res.get('test_pass')} | {dt:.1f}ms | rule={res.get('rule')}")
        sys.stdout.flush()

    print(f"\nBilan Wave 20 : {'TOUS SUCCÈS 20/20' if all_ok else 'DES ÉCHECS DÉTECTÉS'}")

if __name__ == "__main__":
    test_wave20()
