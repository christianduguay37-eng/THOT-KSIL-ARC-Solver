import json
import numpy as np

def solve_9f236235(inp):
    inp = np.array(inp)
    H, W = inp.shape
    
    divider_candidates = {}
    for c in np.unique(inp):
        if c == 0:
            continue
        full_rows = [r for r in range(H) if np.all(inp[r, :] == c)]
        full_cols = [col for col in range(W) if np.all(inp[:, col] == c)]
        if len(full_rows) > 0 and len(full_cols) > 0:
            divider_candidates[c] = (full_rows, full_cols)
            
    if not divider_candidates:
        return inp.copy()
        
    div_c = max(divider_candidates, key=lambda k: len(divider_candidates[k][0]) + len(divider_candidates[k][1]))
    div_rows, div_cols = divider_candidates[div_c]
    
    row_starts = [0] + [r + 1 for r in div_rows]
    row_ends = div_rows + [H]
    col_starts = [0] + [c + 1 for c in div_cols]
    col_ends = div_cols + [W]
    
    n_rows = len(row_starts)
    n_cols = len(col_starts)
    
    macro = np.zeros((n_rows, n_cols), dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            cell = inp[row_starts[i]:row_ends[i], col_starts[j]:col_ends[j]]
            vals = cell[cell != 0]
            if len(vals) > 0:
                macro[i, j] = vals[0]
                
    return np.fliplr(macro)

if __name__ == "__main__":
    with open("training/9f236235.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_9f236235(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_9f236235(inp)
        print(f"Test {i} output shape: {pred.shape}")
        
    if all_ok:
        print("ALL TESTS PASSED for 9f236235!")
