import json
import numpy as np

def solve_c444b776(inp):
    inp = np.array(inp)
    H, W = inp.shape
    out = inp.copy()
    
    div_colors = [c for c in np.unique(inp) if c != 0]
    # Find divider color that forms complete rows or cols
    div_c = 4
    for c in div_colors:
        full_r = [r for r in range(H) if np.all(inp[r, :] == c)]
        full_col = [col for col in range(W) if np.all(inp[:, col] == c)]
        if full_r or full_col:
            div_c = c
            break
            
    div_rows = [r for r in range(H) if np.all(inp[r, :] == div_c)]
    div_cols = [c for c in range(W) if np.all(inp[:, c] == div_c)]
    
    r_splits = [0] + [r + 1 for r in div_rows]
    r_ends = div_rows + [H]
    c_splits = [0] + [c + 1 for c in div_cols]
    c_ends = div_cols + [W]
    
    source_chamber = None
    for r0, r1 in zip(r_splits, r_ends):
        for c0, c1 in zip(c_splits, c_ends):
            sub = inp[r0:r1, c0:c1]
            if np.any((sub != 0) & (sub != div_c)):
                source_chamber = sub.copy()
                break
        if source_chamber is not None:
            break
            
    if source_chamber is not None:
        for r0, r1 in zip(r_splits, r_ends):
            for c0, c1 in zip(c_splits, c_ends):
                out[r0:r1, c0:c1] = source_chamber
                
    return out

if __name__ == "__main__":
    with open("training/c444b776.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_c444b776(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_c444b776(inp)
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
        print("ALL TESTS PASSED for c444b776!")
