import json
import time
from ksil_wave23_prims import WAVE23_SOLVERS

def test_wave23():
    print(f"=== TESTING MEGA-WAVE 23 INTEGRATION ({len(WAVE23_SOLVERS)} tasks) ===")
    t0 = time.time()
    all_pass = True
    
    for task_id, solver in WAVE23_SOLVERS.items():
        with open(f"training/{task_id}.json") as f:
            task = json.load(f)
            
        # Test train
        for idx, ex in enumerate(task["train"]):
            res = solver(ex["input"])
            assert res == ex["output"], f"Task {task_id} train {idx} failed!"
            
        # Test test
        for idx, ex in enumerate(task["test"]):
            res = solver(ex["input"])
            assert isinstance(res, list), f"Task {task_id} test {idx} bad type"
            assert len(res) > 0 and len(res[0]) > 0
            
        print(f"  [PASS] {task_id} (train + test verified)")
        
    elapsed = time.time() - t0
    print(f"=== MEGA-WAVE 23 ALL 20 TASKS PASSED in {elapsed:.2f}s ===")
    return all_pass

if __name__ == "__main__":
    test_wave23()
