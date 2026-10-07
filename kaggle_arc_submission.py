"""
==========================================================================
 🚀 KAGGLE ARC PRIZE : PIPELINE OFFICIEL DE SOUMISSION AUTONOME
==========================================================================
Auteurs : Christian Duguay & Alix (Binôme Souverain)
Projet : ARC Prize (ARC-AGI) / Compétition Officielle Kaggle
Socle Déterministe : Moteur Multiversel THOT-KSIL
==========================================================================
Format Officiel de Sortie (submission.json) :
{
    "task_id": [
        {
            "attempt_1": [[...]],
            "attempt_2": [[...]]
        }
    ]
}
==========================================================================
"""

import os
import sys
import json
import time
import argparse
import glob
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Inclusion du chemin local
sys.path.insert(0, os.path.dirname(__file__))

from thot_arc_core import ArcTask, ArcGrid
from thot_ksil_multiverse import ThotMultiverseSolver

def convert_to_python_ints(obj: Any) -> Any:
    """Garantit que tous les types numpy sont convertis en int Python standards."""
    if isinstance(obj, np.ndarray):
        return obj.astype(int).tolist()
    elif isinstance(obj, list):
        return [convert_to_python_ints(item) for item in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj

def generate_fallback_attempts(input_grid: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Génère deux prédictions heuristiques de secours lorsque le solveur
    ne trouve pas de fermeture déterministe à 100% sur le train :
    - Attempt 1 : Identité stricte (grille d'entrée)
    - Attempt 2 : Découpe de la boîte englobante ou couleur dominante
    """
    attempt_1 = input_grid.astype(int).tolist()
    
    # Attempt 2 : essai de rognage de la boîte englobante des non-zéros
    non_zeros = np.argwhere(input_grid != 0)
    if len(non_zeros) > 0:
        rmin, cmin = non_zeros.min(axis=0)
        rmax, cmax = non_zeros.max(axis=0)
        attempt_2 = input_grid[rmin:rmax+1, cmin:cmax+1].astype(int).tolist()
    else:
        attempt_2 = attempt_1
        
    return attempt_1, attempt_2

def run_submission_pipeline(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Exécute le solveur THOT-KSIL sur les défis et génère le fichier de soumission.
    Supporte :
    1. Un fichier JSON unique (ex: arc-agi_test_challenges.json de Kaggle)
    2. Un répertoire contenant des fichiers .json individuels (ex: evaluation/ ou training/)
    """
    print("=" * 72)
    print(" 🏛️ PIPELINE DE SOUMISSION ARC PRIZE — THOT-KSIL MULTIVERSE")
    print(f" 👉 Source d'entrée : {input_path}")
    print(f" 👉 Sortie cible   : {output_path}")
    print("=" * 72)

    solver = ThotMultiverseSolver()
    submission_data: Dict[str, List[Dict[str, List[List[int]]]]] = {}

    tasks_to_process: List[ArcTask] = []

    t_start = time.time()

    if os.path.isdir(input_path):
        json_files = sorted(glob.glob(os.path.join(input_path, "*.json")))
        print(f"📁 Mode Répertoire détecté : {len(json_files)} tâches trouvées.")
        for f in json_files:
            tasks_to_process.append(ArcTask.load_from_file(f))
    elif os.path.isfile(input_path):
        print("📄 Mode Fichier Unique détecté (Format Challenge Kaggle).")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_challenges = json.load(f)
        for tid, tdata in raw_challenges.items():
            tasks_to_process.append(ArcTask(tid, tdata))
    else:
        raise FileNotFoundError(f"Chemin d'entrée introuvable : {input_path}")

    total_tasks = len(tasks_to_process)
    solved_count = 0
    fallback_count = 0

    print(f"\n⚡ Lancement de l'inférence déterministe sur {total_tasks} tâches...")

    for idx, task in enumerate(tasks_to_process):
        res = solver.solve(task)
        task_id = task.task_id
        task_attempts: List[Dict[str, List[List[int]]]] = []

        if res.get("solved", False) and res.get("predictions"):
            solved_count += 1
            preds = res["predictions"]
            for p_idx, pred in enumerate(preds):
                pred_list = convert_to_python_ints(pred)
                task_attempts.append({
                    "attempt_1": pred_list,
                    "attempt_2": pred_list  # Identique si règle maîtresse prouvée
                })
        else:
            fallback_count += 1
            for p_idx, test_pair in enumerate(task.test_pairs):
                inp = test_pair["input"]
                att1, att2 = generate_fallback_attempts(inp)
                task_attempts.append({
                    "attempt_1": att1,
                    "attempt_2": att2
                })

        submission_data[task_id] = task_attempts

        if (idx + 1) % 50 == 0 or idx == total_tasks - 1:
            elapsed = time.time() - t_start
            print(f"  --> Progression : [{idx+1}/{total_tasks}] ({(idx+1)/total_tasks*100:.1f}%) | "
                  f"Résolues directes : {solved_count} | Temps : {elapsed:.2f}s", flush=True)

    elapsed_total = time.time() - t_start

    # Écriture du fichier submission.json
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission_data, f)

    file_size_kb = os.path.getsize(output_path) / 1024.0

    print("\n" + "=" * 72)
    print(" 🏁 SYNTHÈSE OFFICIELLE DE SOUMISSION ARC PRIZE :")
    print(f"  ✅ Tâches traitées         : {total_tasks}")
    print(f"  🎯 Règle Maîtresse Déduite : {solved_count} ({solved_count/total_tasks*100:.2f}%)")
    print(f"  🛡️ Heuristique de Secours   : {fallback_count} ({fallback_count/total_tasks*100:.2f}%)")
    print(f"  ⏱️ Temps d'exécution total : {elapsed_total:.2f} secondes")
    print(f"  ⚡ Latence moyenne         : {elapsed_total/total_tasks*1000.0:.2f} ms / tâche")
    print(f"  💾 Taille du submission.json: {file_size_kb:.1f} Ko")
    print(f"  📁 Fichier scellé          : {os.path.abspath(output_path)}")
    print("=" * 72)

    return {
        "total_tasks": total_tasks,
        "solved_count": solved_count,
        "fallback_count": fallback_count,
        "elapsed_total": elapsed_total,
        "output_path": output_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de soumission Kaggle ARC Prize - THOT-KSIL")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Chemin vers le fichier JSON de challenge ou le dossier de tâches.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Chemin vers le fichier de sortie submission.json.")
    args = parser.parse_args()

    default_base = os.path.dirname(__file__)
    inp = args.input or os.path.join(default_base, "evaluation")
    out = args.output or os.path.join(default_base, "submission.json")

    run_submission_pipeline(inp, out)
