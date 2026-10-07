import json
import numpy as np

with open("training/846bdb03.json") as f:
    d = json.load(f)

p = d["train"][1]
inp = np.array(p["input"])
exp = np.array(p["output"])

# Let's inspect shape_l and shape_r in inp
inp_no = inp.copy()
inp_no[6:13, 1:9] = 0

print("inp_no[1:6, :]:\n", inp_no[1:6, :])
