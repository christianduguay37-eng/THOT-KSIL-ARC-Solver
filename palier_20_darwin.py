"""
==========================================================================
 🧬 PALIER 20 : LA BOUCLE D'AUTO-ÉVOLUTION DÉTERMINISTE (THOT-DARWIN)
==========================================================================
Auteurs : Christian Duguay & Alix (Binôme Souverain)
Projet : ARC Prize & Moteur K-SIL Multiversel

Principe Fondateur du Palier 20 :
- Observer les énigmes non résolues.
- Extraire leurs invariants géométriques manquants.
- Forger la primitive dans le connectome.
- Valider le critère SRE strict : Zéro Régression sur les 26 tâches certifiées.
- Enregistrer l'avancée et incrémenter le score sans intervention manuelle.
==========================================================================
"""

import os
import sys
import time
import glob
from typing import List, Dict, Tuple, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from thot_arc_core import ArcTask
from thot_ksil_multiverse import ThotMultiverseSolver

# Tâches officiellement certifiées 100% pixel-perfect (Garde-fous anti-régression)
CANONICAL_BENCHMARK_TASKS = [
    "007bbfb7", "00d62c1b", "017c7c7b", "025d127b", "045e512c", "0520fde7", "05269061", "05f2a901",
    "06df4c85", "08ed6ac7", "09629e4f", "0962bcdd", "0a938d79", "0b148d64", "0ca9ddb6",
    "0d3d703e", "0dfd9992", "11852cab", "1190e5a7", "137eaa0f", "178fcbfb", "1a07d186", "1b2d62fb", "1b60fb0c",
    "1bfc4729", "1c786137", "1caeab9d", "1cf80156", "1e0a9b12", "1e32b0e9", "1f0c79e5", "1f642eb9", "1f85a75f",
    "1f876c06", "1fad071e",
    "2013d3e2", "2204b7a8", "22168020", "22233c11", "2281f1f4", "228f6490", "22eb0ac0", "23581191", "239be575", "23b5c85d",
    "253bf280", "25d487eb", "25d8a9c8", "25ff71a9", "27a28665", "28bf18c6", "28e73c20", "29623171", "29c11459", "29ec7d0e", "2bcee788", "2bee17df", "2c608aff", "2dc579da", "2dee498d", "31aa019c", "321b1fc6", "3428a4f5", "3618c87e", "3906de3d", "3aa6fb7a", "3ac3eb23", "3c9b0459", "4258a5f9", "444801d8", "445eab21", "44d8ac46", "44f52bb0", "48d8fb45", "496994bd", "54d82841", "5582e5ca", "5614dbcf", "5bd6f4ac", "5c0a986e", "6150a2bd", "6430c8c4", "67385a82", "6773b310", "67a3c6ac", "67a423a3", "681b3aeb", "68b16354", "6c434453",
    "7468f01a", "74dd1130", "88a62173", "94f9d214", "99b1bc43", "9dfd6313", "a5313dff", "a87f7484", "aabf363d", "aedd82e4", "b1948b0a",
    "be94b721", "c3f564a4", "c8f0f002", "ce4f8723", "d511f180", "dbc1a6ce", "dc1df850", "ded97339", "ed36ccf7", "f2829549", "fafffa47",
    "bc1d5164", "6f8cd79b", "794b24be", "b0c4d837", "b9b7f026", "cdecee7f", "39a8645d", "5117e062", "7ddcd7ec",
    "3f7978a0", "3de23699", "3af2c5a8", "49d1d64f", "46442a0e", "760b3cac", "7447852a", "ae4f1146", "4522001f", "a61ba2ce",
    "62c24649", "67e8384a", "7fe24cdd",
    "6d0aefbc", "6fa7a44f", "4c4377d9", "7b7f7511", "746b3537", "834ec97d", "77fdfe62", "75b8110e", "54d9e175", "72ca375d",
    "8be77c9e", "c9e6f938",
    "8f2ea7aa", "8403a5d5", "868de0fa", "8e5a5113", "60b61512", "88a10436", "694f12f3", "9172f3a0", "963e52fc", "4612dd53",
    "4093f84a", "3bd67248", "3befdf3e", "3eda0437", "3bdb4ada", "363442ee", "4347f46a", "46f33fce", "4be741c5", "5ad4f10b",
    "47c1f68c", "56ff96f3", "508bd3b6", "4c5c2cf0", "53b68214", "50cb2852", "56dc2b01", "41e4d17e", "4938f0c2", "3345333e",
    "662c240a", "6855a6e4", "63613498", "5521c0d9", "5168d44c", "7c008303", "6d75e8bb", "6d0160f0", "6e82a1ae", "623ea044",
    "93b581b8", "8d5021e8", "91413438", "82819916", "952a094c", "941d9a10", "8d510a79", "7f4411dc", "855e0971", "7e0986d6",
    "9565186b", "913fb3ed", "80af3007", "8e1813be", "32597951", "6e02f1e3", "85c4e7cd", "95990924", "810b9b61", "995c5fa3",
    "e9afcf9a", "a79310a0", "bdad9b1f", "d037b0a7", "d631b094", "ac0a08a4", "b91ae062", "c3e719e8", "cce03e0d", "d13f3404",
    "ba97ae07", "e9614598", "b8cdaf2b", "d90796e8", "dae9d2b5", "a416b8f3", "c59eb873", "bbc9ae5d", "d4469b4b", "dc433765",
    "a740d043", "f25ffba3", "a3df8b1e", "a85d4709", "a699fb00", "a9f96cdd", "ff28f65a", "d10ecb37", "cf98881b", "f8b3ba0a",
    "f8ff0b80", "ea786f4a", "d23f8c26", "d89b689b", "d43fd935", "d9fac9be", "de1cd16c", "42a50994", "8efcae92", "e50d258f",
    "99fa7670", "9af7a82c", "f76d97a5", "bda2d7a6", "db3e9e38", "e179c5f4", "ce22a75a", "b60334d2", "a61f2674", "a1570a43",
    "f5b8619d", "f9012d9b", "780d0b14", "7837ac64", "d0f5fe59",
    "a68b268e", "6d58a25d", "cbded52d", "feca6190", "a5f85a15", "bd4472b8", "ba26e723", "a65b410d", "ec883f72", "c9f8e694",
    "e76a88a6", "d8c310e9", "b6afb2da", "d364b489", "caa06a1f", "b230c067", "bb43febb", "c0f76784", "d2abd087", "b548a754",
    "af902bf9", "ce9e57f2", "a48eeaf7", "a3325580", "6e19193c", "beb8660c", "ddf7fa4f", "d6ad076f", "d4f3cd78", "d9f24cd1",
    "e3497940", "97999447", "b94a9452", "e48d4e1a", "d4a91cb9", "673ef223", "d406998b", "d5d6de2d", "a2fd1cf0", "b2862040",
    "90f3ed37", "d06dbe63", "ef135b50", "ea32f347", "f25fbde4", "e98196ab", "a78176bb", "fcc82909", "539a4f51", "d687bc17",
    "e21d9049", "272f95fa", "543a7ed5", "928ad970", "e8593010", "6cdd2623", "b7249182", "eb281b96", "fcb5c309", "f15e1fac",
    "e26a3af2", "ae3edfdc", "e40b9e2f", "f8a8fe49", "c8cbb738", "b190f7f5", "469497ad", "9ecd008a", "91714a58", "90c28cc7",
    "8731374e", "a8c38be5", "eb5a1d5d", "8eb1be9a", "7b6016b9", "83302e8f", "39e1d7f9", "ff805c23", "6b9890af", "ecdecbb3",
    # Mega-Wave 21 (Palier 90.00% - 360/400)
    "10fcaaa3", "150deff5", "234bbc79", "0e206a2e", "264363fd",
    "2dd70a9a", "3631a71a", "36d67576", "36fdfd69", "3e980e27",
    "40853293", "4290ef0e", "447fd412", "484b58aa", "50846271",
    "57aa92db", "5c2c9af4", "5daaa586", "6455b5f5", "6a1e5592",
    # Mega-Wave 22 (Palier 95.00% - 380/400)
    "6aa20dc0", "6cf79266", "6ecd11f4", "72322fa7", "73251a56",
    "776ffc46", "7df24a62", "846bdb03", "890034e9", "8a004b2b",
    "97a05b5b", "98cf29f8", "9aec4887", "9d9215db", "9edfc990",
    "9f236235", "a64e4611", "a8d7556c", "aba27056", "b27ca6d3",
    # Mega-Wave 23 (Cap 100.00% Grand Chelem - 400/400)
    "b527c5c6", "b775ac94", "b782dc8a", "b8825c91", "c1d99e64",
    "c444b776", "c909285e", "ce602527", "d07ae81c", "d22278a0",
    "db93a21d", "dc0a314f", "e5062a87", "e509e548", "e6721834",
    "e73095fd", "e8dc4411", "f1cefba8", "f35d900a", "f8c80d96"
]

class Palier20AutoEvolution:
    """Moteur d'itération et de vérification continue Palier 20."""

    def __init__(self, training_dir: str):
        self.training_dir = training_dir
        self.solver = ThotMultiverseSolver()

    def run_regression_guard(self) -> bool:
        """Vérifie que les 400 tâches canoniques passent toujours à 100% (Cap 100.00% Grand Chelem)."""
        print(f"🛡️ [PALIER 20 - HORUS] Exécution de la suite anti-régression canonique ({len(CANONICAL_BENCHMARK_TASKS)} tâches)...", flush=True)
        t_start = time.time()
        all_passed = True
        for idx, tid in enumerate(CANONICAL_BENCHMARK_TASKS):
            fpath = os.path.join(self.training_dir, f"{tid}.json")
            if not os.path.exists(fpath):
                continue
            task = ArcTask.load_from_file(fpath)
            res = self.solver.solve(task)
            if not (res["solved"] and res["test_pass"]):
                print(f"  ❌ RÉGRESSION DÉTECTÉE sur [{idx+1}/{len(CANONICAL_BENCHMARK_TASKS)}] {tid} !", flush=True)
                all_passed = False
            if (idx + 1) % 50 == 0 or idx == len(CANONICAL_BENCHMARK_TASKS) - 1:
                elapsed = time.time() - t_start
                print(f"  --> Étape {idx+1}/{len(CANONICAL_BENCHMARK_TASKS)} validée | Temps écoulé: {elapsed:.2f}s", flush=True)
        if all_passed:
            elapsed = time.time() - t_start
            print(f"  ✅ SUITE CANONIQUE SCELLÉE : {len(CANONICAL_BENCHMARK_TASKS)}/{len(CANONICAL_BENCHMARK_TASKS)} tests invariants validés (0 Régression) en {elapsed:.2f}s.", flush=True)
        return all_passed

    def audit_unsolved_candidates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Isole les prochaines tâches non résolues pour déduction de primitives."""
        files = sorted(glob.glob(os.path.join(self.training_dir, "*.json")))
        unsolved = []
        print(f"🔍 [AUDIT CANDIDATS] Recherche des prochaines cibles non résolues (limite={limit})...", flush=True)
        for f in files:
            task = ArcTask.load_from_file(f)
            if task.task_id in CANONICAL_BENCHMARK_TASKS:
                continue
            res = self.solver.solve(task)
            if not res["solved"]:
                unsolved.append({
                    "task_id": task.task_id,
                    "in_shape": task.train_pairs[0]["input"].shape,
                    "out_shape": task.train_pairs[0]["output"].shape,
                    "in_colors": set(task.train_pairs[0]["input"].flatten()),
                    "out_colors": set(task.train_pairs[0]["output"].flatten()),
                })
                print(f"  📌 Cible isolée : {task.task_id} ({len(unsolved)}/{limit})", flush=True)
                if len(unsolved) >= limit:
                    break
        return unsolved


if __name__ == "__main__":
    train_path = os.path.join(os.path.dirname(__file__), "training")
    p20 = Palier20AutoEvolution(train_path)
    p20.run_regression_guard()
    candidates = p20.audit_unsolved_candidates(limit=5)
    print("\n🎯 Prochaines cibles pour enrichissement du connectome Palier 20 :")
    for c in candidates:
        print(f"  ➔ Tâche {c['task_id']} : Forme {c['in_shape']} -> {c['out_shape']} | Couleurs {c['in_colors']} -> {c['out_colors']}")
