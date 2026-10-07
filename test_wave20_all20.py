import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import numpy as np

from test_e26a3af2 import solve_e26a3af2
from test_ae3edfdc import solve_ae3edfdc
from test_e40b9e2f import solve_e40b9e2f
from test_f8a8fe49 import solve_f8a8fe49
from test_c8cbb738 import solve_c8cbb738
from test_b190f7f5 import solve_b190f7f5
from test_469497ad import solve_469497ad
from test_9ecd008a import solve_9ecd008a
from test_91714a58 import solve_91714a58
from test_90c28cc7 import solve_90c28cc7
from test_8731374e import solve_8731374e
from test_a8c38be5 import solve_a8c38be5
from test_eb5a1d5d import solve_eb5a1d5d
from test_8eb1be9a import solve_8eb1be9a
from test_7b6016b9 import solve_7b6016b9
from test_83302e8f import solve_83302e8f
from test_39e1d7f9 import solve_39e1d7f9
from test_ff805c23 import solve_ff805c23
from test_6b9890af import solve_6b9890af
from test_ecdecbb3 import solve_ecdecbb3

tasks = [
    ("e26a3af2", solve_e26a3af2),
    ("ae3edfdc", solve_ae3edfdc),
    ("e40b9e2f", solve_e40b9e2f),
    ("f8a8fe49", solve_f8a8fe49),
    ("c8cbb738", solve_c8cbb738),
    ("b190f7f5", solve_b190f7f5),
    ("469497ad", solve_469497ad),
    ("9ecd008a", solve_9ecd008a),
    ("91714a58", solve_91714a58),
    ("90c28cc7", solve_90c28cc7),
    ("8731374e", solve_8731374e),
    ("a8c38be5", solve_a8c38be5),
    ("eb5a1d5d", solve_eb5a1d5d),
    ("8eb1be9a", solve_8eb1be9a),
    ("7b6016b9", solve_7b6016b9),
    ("83302e8f", solve_83302e8f),
    ("39e1d7f9", solve_39e1d7f9),
    ("ff805c23", solve_ff805c23),
    ("6b9890af", solve_6b9890af),
    ("ecdecbb3", solve_ecdecbb3),
]

passed = 0
for tid, fn in tasks:
    with open(f"training/{tid}.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        inp = np.array(p["input"])
        exp = np.array(p["output"])
        pred = fn(inp)
        assert np.array_equal(pred, exp), f"{tid} train {i} failed!"
    for i, p in enumerate(d["test"]):
        inp = np.array(p["input"])
        exp = np.array(p["output"])
        pred = fn(inp)
        assert np.array_equal(pred, exp), f"{tid} test {i} failed!"
    passed += 1
    print(f"✅ [{passed:02d}/20] {tid}: 100% Train & Test PASS")

print("\n🚀 TOUTES LES 20 TÂCHES DE LA MÉGA-VAGUE 20 SONT PARFAITEMENT VALIDÉES !")
