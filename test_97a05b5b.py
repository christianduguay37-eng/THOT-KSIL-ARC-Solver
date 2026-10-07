import json
import numpy as np

def solve_97a05b5b(inp):
    counts = {c: int((inp==c).sum()) for c in np.unique(inp) if c != 0}
    if not counts:
        return None
    c_base = max(counts, key=counts.get)
    H, W = inp.shape
    
    best_board = None
    best_area = 0
    for r0 in range(H):
        for c0 in range(W):
            if inp[r0, c0] != c_base:
                continue
            for r1 in range(H-1, r0 + 3, -1):
                if inp[r1, c0] != c_base:
                    continue
                for c1 in range(W-1, c0 + 3, -1):
                    if inp[r0, c1] != c_base or inp[r1, c1] != c_base:
                        continue
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area <= best_area:
                        continue
                    if not np.all(inp[r0, c0:c1+1] == c_base):
                        continue
                    if not np.all(inp[r1, c0:c1+1] == c_base):
                        continue
                    if not np.all(inp[r0:r1+1, c0] == c_base):
                        continue
                    if not np.all(inp[r0:r1+1, c1] == c_base):
                        continue
                    best_area = area
                    best_board = (r0, c0, r1, c1)
                    
    if not best_board:
        return None
    r0, c0, r1, c1 = best_board
    board = inp[r0:r1+1, c0:c1+1].copy()
    outside = inp.copy()
    outside[r0:r1+1, c0:c1+1] = 0
    
    pieces = []
    used_mask = np.zeros(inp.shape, dtype=bool)
    for r in range(H - 2):
        for c in range(W - 2):
            if np.all(outside[r:r+3, c:c+3] != 0) and not np.any(used_mask[r:r+3, c:c+3]):
                pieces.append(outside[r:r+3, c:c+3].copy())
                used_mask[r:r+3, c:c+3] = True
                
    H_b, W_b = board.shape
    piece_placements = []
    for piece in pieces:
        placements = []
        for dr in range(H_b - 2):
            for dc in range(W_b - 2):
                B = board[dr:dr+3, dc:dc+3]
                if np.sum(B == 0) == 0:
                    continue
                for rot in range(4):
                    for flip in [False, True]:
                        P = np.rot90(piece, rot)
                        if flip:
                            P = np.fliplr(P)
                        if np.array_equal((P == c_base), (B == 0)):
                            placements.append((dr, dc, P))
        unique_placements = []
        for dr, dc, P in placements:
            if not any(dr == u[0] and dc == u[1] and np.array_equal(P, u[2]) for u in unique_placements):
                unique_placements.append((dr, dc, P))
        piece_placements.append(unique_placements)
        
    solutions = []
    def search(idx, current_canvas, occupied):
        if idx == len(pieces):
            if np.all(current_canvas != 0):
                penalty = np.sum(current_canvas[0, :] != c_base)
                solutions.append((penalty, current_canvas))
            return
        for dr, dc, P in piece_placements[idx]:
            overlap = False
            for odr, odc in occupied:
                if abs(dr - odr) < 3 and abs(dc - odc) < 3:
                    overlap = True
                    break
            if overlap:
                continue
            new_canvas = current_canvas.copy()
            new_canvas[dr:dr+3, dc:dc+3] = P
            search(idx + 1, new_canvas, occupied + [(dr, dc)])
            
    search(0, board.copy(), [])
    solutions.sort(key=lambda x: x[0])
    return solutions[0][1] if solutions else None

if __name__ == "__main__":
    with open("training/97a05b5b.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        assert np.array_equal(solve_97a05b5b(np.array(p["input"])), np.array(p["output"]))
    for i, p in enumerate(d["test"]):
        assert np.array_equal(solve_97a05b5b(np.array(p["input"])), np.array(p["output"]))
    print("97a05b5b: 100% PASS ON ALL TRAIN AND TEST!")
