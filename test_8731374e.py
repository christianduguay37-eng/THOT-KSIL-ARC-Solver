import json
import numpy as np

def solve_8731374e(inp):
    H, W = inp.shape
    best_area = 0
    best_rect = None
    best_bg, best_fg = None, None
    for r0 in range(H):
        for r1 in range(r0 + 2, H):
            for c0 in range(W):
                for c1 in range(c0 + 2, W):
                    sub = inp[r0:r1+1, c0:c1+1]
                    colors = set(np.unique(sub))
                    if len(colors) == 2 and 0 not in colors:
                        area = sub.size
                        if area > best_area:
                            best_area = area
                            c_list = list(colors)
                            bg = c_list[0] if (sub == c_list[0]).sum() > (sub == c_list[1]).sum() else c_list[1]
                            fg = c_list[1] if bg == c_list[0] else c_list[0]
                            best_rect = (r0, r1, c0, c1)
                            best_bg, best_fg = bg, fg
                            
    r0, r1, c0, c1 = best_rect
    h, w = r1 - r0 + 1, c1 - c0 + 1
    sub = inp[r0:r1+1, c0:c1+1]
    out = np.full((h, w), best_bg, dtype=int)
    for pr, pc in np.argwhere(sub == best_fg):
        out[pr, :] = best_fg
        out[:, pc] = best_fg
    return out

with open("training/8731374e.json") as f:
    d = json.load(f)

for i, p in enumerate(d["train"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_8731374e(inp)
    assert np.array_equal(pred, expected), f"Train {i} failed!"
for i, p in enumerate(d["test"]):
    inp = np.array(p["input"])
    expected = np.array(p["output"])
    pred = solve_8731374e(inp)
    assert np.array_equal(pred, expected), f"Test {i} failed!"
print("8731374e: 100% PASS ON ALL TRAIN AND TEST!")
