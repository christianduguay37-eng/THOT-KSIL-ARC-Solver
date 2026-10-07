import json
import numpy as np

def solve_36fdfd69(grid):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    
    pts2 = list(zip(*np.where(grid == 2)))
    if not pts2:
        return grid
        
    # Cluster 2s with Chebyshev distance <= 2
    N = len(pts2)
    parent = list(range(N))
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            
    for i in range(N):
        r1, c1 = pts2[i]
        for j in range(i + 1, N):
            r2, c2 = pts2[j]
            if max(abs(r1 - r2), abs(c1 - c2)) <= 2:
                union(i, j)
                
    clusters = {}
    for i in range(N):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(pts2[i])
        
    out = grid.copy()
    for root, pts in clusters.items():
        min_r = min(r for r, c in pts)
        max_r = max(r for r, c in pts)
        min_c = min(c for r, c in pts)
        max_c = max(c for r, c in pts)
        
        # In this bounding box, change any non-zero, non-2 cell to 4
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if out[r, c] != 0 and out[r, c] != 2:
                    out[r, c] = 4
                    
    return out

if __name__ == "__main__":
    with open("training/36fdfd69.json") as f:
        d = json.load(f)
    for idx, p in enumerate(d["train"]):
        res = solve_36fdfd69(p["input"])
        expected = np.array(p["output"])
        assert np.array_equal(res, expected), f"Train {idx} failed"
    for idx, p in enumerate(d["test"]):
        res = solve_36fdfd69(p["input"])
        if "output" in p:
            expected = np.array(p["output"])
            assert np.array_equal(res, expected), f"Test {idx} failed"
    print("ALL TESTS PASSED FOR 36fdfd69!")
