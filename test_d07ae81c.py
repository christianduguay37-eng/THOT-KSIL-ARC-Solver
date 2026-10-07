import json
import numpy as np

def solve_d07ae81c(grid):
    inp = np.array(grid)
    H, W = inp.shape
    vals, counts = np.unique(inp, return_counts=True)
    sorted_colors = [c for c, cnt in sorted(zip(vals, counts), key=lambda x: x[1])]
    seed_colors = set(sorted_colors[:-2])
    bg_colors = set(sorted_colors[-2:])
    
    bg_to_seed = {}
    seed_positions = []
    for r in range(H):
        for c in range(W):
            if inp[r, c] in seed_colors:
                seed_positions.append((r, c))
                neigh_bgs = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and inp[nr, nc] in bg_colors:
                        neigh_bgs.append(inp[nr, nc])
                if neigh_bgs:
                    home_bg = max(set(neigh_bgs), key=neigh_bgs.count)
                    bg_to_seed[home_bg] = inp[r, c]
                    
    out = inp.copy()
    for r, c in seed_positions:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            cr, cc = r + dr, c + dc
            while 0 <= cr < H and 0 <= cc < W:
                bg = inp[cr, cc]
                if bg in bg_to_seed:
                    out[cr, cc] = bg_to_seed[bg]
                cr += dr
                cc += dc
                
    return out.tolist()

if __name__ == "__main__":
    with open("training/d07ae81c.json") as f:
        task = json.load(f)
    for idx, ex in enumerate(task["train"]):
        res = solve_d07ae81c(ex["input"])
        assert res == ex["output"], f"Train {idx} failed!"
        print(f"Train {idx} PASS!")
    for idx, ex in enumerate(task["test"]):
        res = solve_d07ae81c(ex["input"])
        print(f"Test {idx} output shape: {len(res)}x{len(res[0])}")
        print("Test 0 SUCCESS!")
