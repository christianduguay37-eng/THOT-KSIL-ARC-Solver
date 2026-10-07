import json
import numpy as np
from thot_arc_core import ArcTask
from thot_ksil_multiverse import ThotMultiverseSolver
from ksil_wave23_prims import get_wave23_primitives

solver = ThotMultiverseSolver()
prims = get_wave23_primitives()

for tid in ['e6721834', 'e73095fd', 'f8c80d96']:
    task = ArcTask.load_from_file(f'training/{tid}.json')
    print(f'=== Testing {tid} ===')
    for pname, pfn in prims.items():
        match = True
        for p in task.train_pairs:
            pred = pfn(p['input'])
            if pred is None or not np.array_equal(pred, p['output']):
                match = False
                break
        if match:
            print(f'  Matching primitive found: {pname}')
            test_inp = task.test_pairs[0]['input']
            test_pred = pfn(test_inp)
            print(f'  test pred is None? {test_pred is None}')
            
    res = solver.solve(task)
    print(f'  solver.solve: solved={res["solved"]}, test_pass={res.get("test_pass")}, rule={res.get("rule")}')
