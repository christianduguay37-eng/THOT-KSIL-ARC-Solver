import json
import numpy as np

def solve_9d9215db(inp):
    inp = np.array(inp)
    H, W = inp.shape
    out = np.zeros((H, W), dtype=int)
    
    # Determine the odd-grid step (usually 2, with offset 1)
    # Check all square shells s
    possible_s = [s for s in range(1, min(H, W) // 2 + 1, 2)]
    
    for s in possible_s:
        corners = [(s, s), (s, W - 1 - s), (H - 1 - s, s), (H - 1 - s, W - 1 - s)]
        c_corner = 0
        active_corner = None
        for r, c in corners:
            if inp[r, c] != 0:
                c_corner = inp[r, c]
                active_corner = (r, c)
                break
                
        if c_corner == 0:
            continue
            
        for r, c in corners:
            out[r, c] = c_corner
            
        r0, c0 = active_corner
        opp_c = W - 1 - c0
        opp_r = H - 1 - r0
        
        dc = 2 if opp_c > c0 else -2
        dr = 2 if opp_r > r0 else -2
        
        p_row = (r0, c0 + dc)
        p_col = (r0 + dr, c0)
        
        c_edge = 0
        if 0 <= p_row[0] < H and 0 <= p_row[1] < W and inp[p_row] != 0:
            c_edge = inp[p_row]
        elif 0 <= p_col[0] < H and 0 <= p_col[1] < W and inp[p_col] != 0:
            c_edge = inp[p_col]
            
        if c_edge != 0:
            for c in range(s + 2, W - 1 - s, 2):
                out[s, c] = c_edge
                out[H - 1 - s, c] = c_edge
            for r in range(s + 2, H - 1 - s, 2):
                out[r, s] = c_edge
                out[r, W - 1 - s] = c_edge
                
    return out

if __name__ == "__main__":
    with open("training/9d9215db.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_9d9215db(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_9d9215db(inp)
        print(f"Test {i} output shape: {pred.shape}")
        
    if all_ok:
        print("ALL TESTS PASSED for 9d9215db!")
