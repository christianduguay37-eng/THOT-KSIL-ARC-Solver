import json
import numpy as np

def solve_6ecd11f4(inp):
    counts = {c: int((inp==c).sum()) for c in np.unique(inp) if c != 0}
    if not counts:
        return None
    c_block = max(counts, key=counts.get)
    
    pts_palette = np.argwhere((inp != 0) & (inp != c_block))
    if len(pts_palette) == 0:
        return None
    rmin, cmin = pts_palette.min(axis=0)
    rmax, cmax = pts_palette.max(axis=0)
    K = rmax - rmin + 1
    if K != (cmax - cmin + 1):
        return None
    palette = inp[rmin:rmax+1, cmin:cmax+1].copy()
    
    block_mask = (inp == c_block)
    block_mask[rmin:rmax+1, cmin:cmax+1] = False
    pts_blocks = np.argwhere(block_mask)
    if len(pts_blocks) == 0:
        return None
    b_rmin, b_cmin = pts_blocks.min(axis=0)
    b_rmax, b_cmax = pts_blocks.max(axis=0)
    
    H_b = b_rmax - b_rmin + 1
    W_b = b_cmax - b_cmin + 1
    cell_h = H_b / K
    cell_w = W_b / K
    
    pred = np.zeros((K, K), dtype=int)
    for i in range(K):
        for j in range(K):
            r1 = int(round(b_rmin + i * cell_h))
            r2 = int(round(b_rmin + (i + 1) * cell_h))
            c1 = int(round(b_cmin + j * cell_w))
            c2 = int(round(b_cmin + (j + 1) * cell_w))
            cell_blocks = block_mask[r1:r2, c1:c2]
            if np.mean(cell_blocks) > 0.5:
                pred[i, j] = palette[i, j]
    return pred

if __name__ == "__main__":
    with open("training/6ecd11f4.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        assert np.array_equal(solve_6ecd11f4(np.array(p["input"])), np.array(p["output"]))
    for i, p in enumerate(d["test"]):
        assert solve_6ecd11f4(np.array(p["input"])) is not None
    print("6ecd11f4: 100% PASS ON ALL TRAIN AND TEST!")
