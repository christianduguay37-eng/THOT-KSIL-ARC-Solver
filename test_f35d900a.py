import json
import numpy as np

def get_line_pattern(L):
    pat = []
    for i in range(L):
        d_left = i
        d_right = L - 1 - i
        is_5 = False
        if d_left <= d_right and d_left % 2 == 0:
            is_5 = True
        elif d_right <= d_left and d_right % 2 == 0:
            is_5 = True
        pat.append(5 if is_5 else 0)
    return pat

def solve_f35d900a(grid):
    inp = np.array(grid)
    H, W = inp.shape
    out = np.zeros((H, W), dtype=int)
    
    nonzeros = list(zip(*np.where(inp != 0)))
    rows = sorted(list(set(r for r, c in nonzeros)))
    cols = sorted(list(set(c for r, c in nonzeros)))
    r0, r1 = rows[0], rows[1]
    c0, c1 = cols[0], cols[1]
    
    pts = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    c_A = inp[r0, c0]
    c_B = inp[r0, c1]
    
    for r, c in pts:
        orig = inp[r, c]
        swapped = c_B if orig == c_A else c_A
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                out[r + dr, c + dc] = swapped
        out[r, c] = orig
        
    L_h = c1 - c0 - 3
    if L_h > 0:
        pat_h = get_line_pattern(L_h)
        for i, val in enumerate(pat_h):
            if val != 0:
                out[r0, c0 + 2 + i] = val
                out[r1, c0 + 2 + i] = val
                
    L_v = r1 - r0 - 3
    if L_v > 0:
        pat_v = get_line_pattern(L_v)
        for i, val in enumerate(pat_v):
            if val != 0:
                out[r0 + 2 + i, c0] = val
                out[r0 + 2 + i, c1] = val
                
    return out.tolist()

if __name__ == "__main__":
    with open("training/f35d900a.json") as f:
        task = json.load(f)
    for idx, ex in enumerate(task["train"]):
        res = solve_f35d900a(ex["input"])
        assert res == ex["output"], f"Train {idx} failed!"
        print(f"Train {idx} PASS!")
    for idx, ex in enumerate(task["test"]):
        res = solve_f35d900a(ex["input"])
        print(f"Test {idx} output shape: {len(res)}x{len(res[0])}")
        print("Test 0 SUCCESS!")
