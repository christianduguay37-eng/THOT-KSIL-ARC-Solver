import json
import numpy as np

from test_10fcaaa3 import solve_10fcaaa3
from test_150deff5 import solve_150deff5
from test_234bbc79 import solve_234bbc79
from test_0e206a2e import solve_0e206a2e
from test_264363fd import solve_264363fd
from test_2dd70a9a import solve_2dd70a9a
from test_3631a71a import solve_3631a71a
from test_36d67576 import solve_36d67576
from test_36fdfd69 import solve_36fdfd69
from test_3e980e27 import solve_3e980e27
from test_40853293 import solve_40853293
from test_4290ef0e import solve_4290ef0e
from test_447fd412 import solve_447fd412
from test_484b58aa import solve_484b58aa
from test_50846271 import solve_50846271
from test_57aa92db import solve_57aa92db
from test_5c2c9af4 import solve_5c2c9af4
from test_5daaa586 import solve_5daaa586
from test_6455b5f5 import solve_6455b5f5
from test_6a1e5592 import solve_6a1e5592

WAVE21_SOLVERS = {
    "10fcaaa3": solve_10fcaaa3,
    "150deff5": solve_150deff5,
    "234bbc79": solve_234bbc79,
    "0e206a2e": solve_0e206a2e,
    "264363fd": solve_264363fd,
    "2dd70a9a": solve_2dd70a9a,
    "3631a71a": solve_3631a71a,
    "36d67576": solve_36d67576,
    "36fdfd69": solve_36fdfd69,
    "3e980e27": solve_3e980e27,
    "40853293": solve_40853293,
    "4290ef0e": solve_4290ef0e,
    "447fd412": solve_447fd412,
    "484b58aa": solve_484b58aa,
    "50846271": solve_50846271,
    "57aa92db": solve_57aa92db,
    "5c2c9af4": solve_5c2c9af4,
    "5daaa586": solve_5daaa586,
    "6455b5f5": solve_6455b5f5,
    "6a1e5592": solve_6a1e5592,
}

def test_wave21_all():
    print(f"=== TESTING MEGA-WAVE 21 ({len(WAVE21_SOLVERS)} TASKS) ===")
    all_passed = True
    for task_id, solver in WAVE21_SOLVERS.items():
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
            if "output" in p:
                expected = np.array(p["output"])
                if not np.array_equal(res, expected):
                    print(f"FAIL: {task_id} test {idx}")
                    task_passed = False
                    all_passed = False
        if task_passed:
            print(f"PASS: {task_id} (100% exact match)")
            
    assert all_passed, "Some tasks failed in Mega-Wave 21!"
    print("ALL 20 TASKS IN MEGA-WAVE 21 PASSED 100% PERFECT!")

if __name__ == "__main__":
    test_wave21_all()
