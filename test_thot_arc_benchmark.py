"""
==========================================================================
 🧪 BANC D'ESSAI THOT ARC BENCHMARK
==========================================================================
Auteurs : Christian Duguay & Alix
Teste le solveur THOT sur les tâches officielles de l'ARC Prize.
==========================================================================
"""

import os
import sys
import time
import glob

# Forcer UTF-8 sur la sortie console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from thot_arc_core import ArcTask, ThotArcSolver

def run_benchmark(dataset_dir: str, max_tasks: int = 50):
    solver = ThotArcSolver()
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))[:max_tasks]
    
    print(f"🚀 Lancement du banc d'essai THOT sur {len(files)} tâches de {dataset_dir}...")
    start_time = time.perf_counter()
    
    solved_count = 0
    test_pass_count = 0
    results_detail = []
    
    for idx, fpath in enumerate(files):
        task = ArcTask.load_from_file(fpath)
        t0 = time.perf_counter()
        res = solver.solve(task)
        dt = (time.perf_counter() - t0) * 1000 # ms
        
        if res["solved"]:
            solved_count += 1
            if res["test_pass"]:
                test_pass_count += 1
            print(f"  ✅ [{idx+1:02d}/{len(files):02d}] Tâche {task.task_id} : RÉSOLUE par « {res['rule']} » ({dt:.2f} ms) | Test Pass: {res['test_pass']}")
            results_detail.append((task.task_id, res['rule'], dt, True))
        else:
            # print(f"  ❌ [{idx+1:02d}/{len(files):02d}] Tâche {task.task_id} : Non résolue ({dt:.2f} ms)")
            results_detail.append((task.task_id, "None", dt, False))
            
    total_time = time.perf_counter() - start_time
    
    print("\n" + "=" * 60)
    print(" 📊 BILAN DU BANC D'ESSAI THOT ARC CORE")
    print("=" * 60)
    print(f"🔹 Tâches testées       : {len(files)}")
    print(f"🔹 Tâches résolues (Train 100%) : {solved_count} ({solved_count / len(files) * 100:.1f} %)")
    print(f"🔹 Test exact match (100% pixel perfect) : {test_pass_count} ({test_pass_count / len(files) * 100:.1f} %)")
    print(f"⚡ Temps total calcul   : {total_time:.3f} s (Moyenne : {total_time / len(files) * 1000:.2f} ms / tâche)")
    print("=" * 60)

if __name__ == "__main__":
    train_dir = os.path.join(os.path.dirname(__file__), "training")
    run_benchmark(train_dir, max_tasks=100)
