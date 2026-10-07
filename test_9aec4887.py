import json
import numpy as np

def solve_9aec4887(inp):
    inp = np.array(inp)
    r8, c8 = np.where(inp == 8)
    if len(r8) == 0:
        return inp.copy()
    pat8 = inp[r8.min():r8.max()+1, c8.min():c8.max()+1].copy()
    h, w = pat8.shape
    
    wall_r, wall_c = np.where((inp != 0) & (inp != 8))
    if len(wall_r) == 0:
        return inp.copy()
    min_r, max_r = wall_r.min(), wall_r.max()
    min_c, max_c = wall_c.min(), wall_c.max()
    
    frame = inp[min_r:max_r+1, min_c:max_c+1].copy()
    
    top_color = frame[0, 1]
    bot_color = frame[-1, 1]
    left_color = frame[1, 0]
    right_color = frame[1, -1]
    
    interior = pat8.copy()
    for r in range(h):
        for c in range(w):
            if interior[r, c] == 8:
                d_top = r
                d_bot = (h - 1) - r
                d_left = c
                d_right = (w - 1) - c
                
                dists = [('top', d_top, top_color), ('bot', d_bot, bot_color), ('left', d_left, left_color), ('right', d_right, right_color)]
                dists.sort(key=lambda x: x[1])
                
                # Takes color of uniquely closest wall; ties remain 8
                if dists[0][1] < dists[1][1]:
                    interior[r, c] = dists[0][2]
                    
    frame[1:-1, 1:-1] = interior
    return frame

if __name__ == "__main__":
    with open("training/9aec4887.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_9aec4887(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_9aec4887(inp)
        if "output" in ex:
            out = np.array(ex["output"])
            if not np.array_equal(pred, out):
                print(f"FAILED Test {i}")
                all_ok = False
            else:
                print(f"PASS Test {i}")
        else:
            print(f"Test {i} output shape: {pred.shape}")
            
    if all_ok:
        print("ALL TESTS PASSED for 9aec4887!")
