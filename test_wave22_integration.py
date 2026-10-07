import json
import numpy as np

from test_6aa20dc0 import solve_6aa20dc0
from test_6cf79266 import solve_6cf79266
from test_6ecd11f4 import solve_6ecd11f4
from test_72322fa7 import solve_72322fa7
from test_73251a56 import solve_73251a56
from test_776ffc46 import solve_776ffc46
from test_7df24a62 import solve_7df24a62
from test_846bdb03 import solve_846bdb03
from test_890034e9 import solve_890034e9
from test_8a004b2b import solve_8a004b2b
from test_97a05b5b import solve_97a05b5b
from test_98cf29f8 import solve_98cf29f8
from test_9aec4887 import solve_9aec4887
from test_9d9215db import solve_9d9215db
from test_9edfc990 import solve_9edfc990
from test_9f236235 import solve_9f236235
from test_a64e4611 import solve_a64e4611
from test_a8d7556c import solve_a8d7556c
from test_aba27056 import solve_aba27056
from test_b27ca6d3 import solve_b27ca6d3

WAVE22_SOLVERS = {
    "6aa20dc0": solve_6aa20dc0,
    "6cf79266": solve_6cf79266,
    "6ecd11f4": solve_6ecd11f4,
    "72322fa7": solve_72322fa7,
    "73251a56": solve_73251a56,
    "776ffc46": solve_776ffc46,
    "7df24a62": solve_7df24a62,
    "846bdb03": solve_846bdb03,
    "890034e9": solve_890034e9,
    "8a004b2b": solve_8a004b2b,
    "97a05b5b": solve_97a05b5b,
    "98cf29f8": solve_98cf29f8,
    "9aec4887": solve_9aec4887,
    "9d9215db": solve_9d9215db,
    "9edfc990": solve_9edfc990,
    "9f236235": solve_9f236235,
    "a64e4611": solve_a64e4611,
    "a8d7556c": solve_a8d7556c,
    "aba27056": solve_aba27056,
    "b27ca6d3": solve_b27ca6d3,
}

def test_wave22_all():
    print(f"=== TESTING MEGA-WAVE 22 ({len(WAVE22_SOLVERS)} TASKS) ===")
    all_passed = True
    for task_id, solver in WAVE22_SOLVERS.items():
        with open(f"training/{task_id}.json") as f:
            d = json.load(f)
        task_passed = True
        for idx, p in enumerate(d["train"]):
            res = solver(np.array(p["input"]))
            expected = np.array(p["output"])
            if not np.array_equal(res, expected):
                print(f"FAIL: {task_id} train {idx}")
                task_passed = False
                all_passed = False
        for idx, p in enumerate(d["test"]):
            res = solver(np.array(p["input"]))
            if res is None or res.shape[0] == 0:
                print(f"FAIL: {task_id} test {idx} returned empty")
                task_passed = False
                all_passed = False
            else:
                if "output" in p:
                    expected = np.array(p["output"])
                    if not np.array_equal(res, expected):
                        print(f"FAIL: {task_id} test {idx}")
                        task_passed = False
                        all_passed = False
        if task_passed:
            print(f"PASS: {task_id} (100% train + test)")
            
    if all_passed:
        print("\nALL 20 TASKS IN MEGA-WAVE 22 PASSED WITH 100% DETERMINISTIC FIDELITY!")
    else:
        print("\nSOME TASKS FAILED.")

if __name__ == "__main__":
    test_wave22_all()
