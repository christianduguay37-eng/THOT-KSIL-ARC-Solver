import json
import itertools
import numpy as np

def solve_4290ef0e(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    
    colors, counts = np.unique(grid, return_counts=True)
    bg = colors[np.argmax(counts)]
    other_colors = [c for c in colors if c != bg]
    
    color_info = []
    for c in other_colors:
        pts = list(zip(*np.where(grid == c)))
        min_r = min(r for r, c_ in pts)
        max_r = max(r for r, c_ in pts)
        min_c = min(c_ for r, c_ in pts)
        max_c = max(c_ for r, c_ in pts)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        max_dim = max(h, w)
        
        sub = grid[min_r:max_r+1, min_c:max_c+1]
        max_run = 0
        for r in range(sub.shape[0]):
            cur = 0
            for v in sub[r, :]:
                if v == c:
                    cur += 1
                    if cur > max_run:
                        max_run = cur
                else:
                    cur = 0
        for col_idx in range(sub.shape[1]):
            cur = 0
            for v in sub[:, col_idx]:
                if v == c:
                    cur += 1
                    if cur > max_run:
                        max_run = cur
                else:
                    cur = 0
                
        color_info.append({
            'color': c,
            'pts': pts,
            'count': len(pts),
            'max_dim': max_dim,
            'max_run': max_run
        })
        
    center_color = bg
    singletons = [ci for ci in color_info if ci['count'] == 1 and ci['max_dim'] == 1]
    if singletons:
        center_color = singletons[0]['color']
        color_info = [ci for ci in color_info if ci != singletons[0]]
        
    M = len(color_info)
    N = 2 * M + 1
    
    best_perm = None
    best_score = float('inf')
    for perm in itertools.permutations(range(1, M + 1)):
        valid = True
        score = 0
        for i, k in enumerate(perm):
            ci = color_info[i]
            if k > 1 and k < ci['max_run']:
                valid = False
                break
            if ci['max_dim'] > 2 * k + 1:
                valid = False
                break
            target_k = max(ci['max_run'], (ci['max_dim'] - 1) // 2)
            score += abs(k - target_k)
        if valid and score < best_score:
            best_score = score
            best_perm = perm
            
    if best_perm is None:
        best_perm = tuple(range(1, M + 1))
        
    quad = np.full((M + 1, M + 1), bg, dtype=int)
    quad[M, M] = center_color
    
    for i, k in enumerate(best_perm):
        ci = color_info[i]
        c = ci['color']
        if k == 1:
            L = 2
        else:
            L = min(ci['max_run'], k)
        for offset in range(L):
            if M - k + offset <= M:
                quad[M - k, M - k + offset] = c
                quad[M - k + offset, M - k] = c
                
    out = np.zeros((N, N), dtype=int)
    for r in range(M + 1):
        for c in range(M + 1):
            val = quad[r, c]
            out[r, c] = val
            out[r, N - 1 - c] = val
            out[N - 1 - r, c] = val
            out[N - 1 - r, N - 1 - c] = val
            
    return out

if __name__ == "__main__":
    with open("training/4290ef0e.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_4290ef0e(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_4290ef0e(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 4290ef0e!")
