import time, sys
from thot_arc_core import ArcTask
from thot_ksil_multiverse import ThotMultiverseSolver
from palier_20_darwin import CANONICAL_BENCHMARK_TASKS

solver = ThotMultiverseSolver()
total = len(CANONICAL_BENCHMARK_TASKS)
print(f"=== DEMARRAGE DU CONTROLE SRE DARWIN ({total} TACHES) ===", flush=True)

start_time = time.time()
passed = 0
failed = []
slow = []

for idx, tid in enumerate(CANONICAL_BENCHMARK_TASKS):
    t0 = time.time()
    try:
        task = ArcTask.load_from_file(f"training/{tid}.json")
        res = solver.solve(task)
        dt = time.time() - t0
        if dt > 0.5:
            slow.append((tid, dt))
        if res.get("solved") and res.get("test_pass"):
            passed += 1
        else:
            failed.append(tid)
            print(f"FAILED on [{idx+1}/{total}] {tid} (dt={dt:.3f}s)", flush=True)
    except Exception as e:
        dt = time.time() - t0
        failed.append(tid)
        print(f"ERROR on [{idx+1}/{total}] {tid}: {e} (dt={dt:.3f}s)", flush=True)
        
    if (idx + 1) % 50 == 0 or idx == total - 1:
        elapsed = time.time() - start_time
        print(f"--> Etape {idx+1}/{total} | Passed: {passed} | Failed: {len(failed)} | Temps ecoule: {elapsed:.2f}s", flush=True)

total_time = time.time() - start_time
print(f"\n==========================================", flush=True)
print(f"RESULTAT FINAL : {passed} / {total} ({(passed/total)*100:.2f}%)", flush=True)
print(f"DUREE TOTALE   : {total_time:.2f}s (Moyenne: {total_time/total:.3f}s/tache)", flush=True)
if failed:
    print(f"TACHES ECHOUEES ({len(failed)}) : {failed}", flush=True)
else:
    print("ZERO REGRESSION : 100% DE SUCCES SUR LES 360 TACHES !", flush=True)
if slow:
    print(f"Taches >0.5s ({len(slow)}) : {slow[:10]}", flush=True)
print(f"==========================================", flush=True)
