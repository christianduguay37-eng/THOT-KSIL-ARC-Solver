import json
import numpy as np

def solve_ce602527(grid):
    inp = np.array(grid)
    vals, counts = np.unique(inp, return_counts=True)
    bg = vals[np.argmax(counts)]
    fg_colors = [c for c in vals if c != bg]
    
    # Each color defines a binary mask
    masks = {}
    for c in fg_colors:
        c_mask = (inp == c)
        rows, cols = np.where(c_mask)
        sub = c_mask[rows.min():rows.max()+1, cols.min():cols.max()+1]
        masks[c] = (sub, rows.min(), rows.max()+1, cols.min(), cols.max()+1)
        
    for c in fg_colors:
        sub, r0, r1, c0, c1 = masks[c]
        sub2 = np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1)
        h2, w2 = sub2.shape
        
        for c_other in fg_colors:
            if c_other == c:
                continue
            osub, _, _, _, _ = masks[c_other]
            oh, ow = osub.shape
            if oh <= h2 and ow <= w2:
                for dr in range(h2 - oh + 1):
                    for dc in range(w2 - ow + 1):
                        if np.array_equal(sub2[dr:dr+oh, dc:dc+ow], osub):
                            # Found match! Return bounding box of c in inp
                            return inp[r0:r1, c0:c1].tolist()
                            
    raise ValueError("No scaled match found")

if __name__ == "__main__":
    with open("training/ce602527.json") as f:
        task = json.load(f)
    for idx, ex in enumerate(task["train"]):
        res = solve_ce602527(ex["input"])
        assert res == ex["output"], f"Train {idx} failed!"
        print(f"Train {idx} PASS!")
    for idx, ex in enumerate(task["test"]):
        res = solve_ce602527(ex["input"])
        print(f"Test {idx} output shape: {len(res)}x{len(res[0])}")
        print("Test 0 SUCCESS!")
