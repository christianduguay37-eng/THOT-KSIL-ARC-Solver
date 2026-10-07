import os
import sys
import time
import glob

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from thot_arc_core import ArcTask
from thot_ksil_multiverse import ThotMultiverseSolver

def run_multiverse_benchmark(dataset_dir: str, max_tasks: int = 400):
    solver = ThotMultiverseSolver()
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))[:max_tasks]
    
    print(f"🌌 Lancement du Moteur THOT K-SIL MULTIVERSE (Palier 19 + VERALUME) sur {len(files)} tâches...")
    start_time = time.perf_counter()
    
    solved_count = 0
    test_pass_count = 0
    multiverse_composed_count = 0
    
    for idx, fpath in enumerate(files):
        task = ArcTask.load_from_file(fpath)
        res = solver.solve(task)
        
        if res["solved"]:
            solved_count += 1
            if res["test_pass"]:
                test_pass_count += 1
            if "➔" in res["rule"]:
                multiverse_composed_count += 1
                tag = "🌀 MULTIVERS (Palier 19)"
            else:
                tag = "⚡ MONO-ÉTAPE"
            print(f"  ✅ [{idx+1:03d}/{len(files):03d}] {task.task_id} : {tag} | Règle: « {res['rule']} » ({res['dt_ms']:.2f} ms) | Test Pass: {res['test_pass']}")
            
    total_time = time.perf_counter() - start_time
    
    print("\n" + "=" * 65)
    print(" 🌌 BILAN DU BANC D'ESSAI THOT K-SIL MULTIVERSE (PALIER 19)")
    print("=" * 65)
    print(f"🔹 Tâches testées                  : {len(files)}")
    print(f"🔹 Tâches résolues (Train 100%)    : {solved_count} ({solved_count / len(files) * 100:.1f} %)")
    print(f"🔹 Test exact match (Pixel Perfect): {test_pass_count} ({test_pass_count / len(files) * 100:.1f} %)")
    print(f"🌀 Résolutions par Branching Multiversel : {multiverse_composed_count}")
    print(f"⚡ Temps total calcul              : {total_time:.3f} s (Moyenne : {total_time / len(files) * 1000:.2f} ms / tâche)")
    print("=" * 65)

if __name__ == "__main__":
    train_dir = os.path.join(os.path.dirname(__file__), "training")
    run_multiverse_benchmark(train_dir, max_tasks=400)
