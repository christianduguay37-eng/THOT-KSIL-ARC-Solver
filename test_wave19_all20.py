import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from collections import defaultdict
from scipy.ndimage import label, binary_dilation

# 1. 90f3ed37
def solve_90f3ed37(inp):
    out = inp.copy()
    rows_with_8 = [r for r in range(inp.shape[0]) if 8 in inp[r, :]]
    if not rows_with_8:
        return out
    groups = []
    curr = [rows_with_8[0]]
    for r in rows_with_8[1:]:
        if r - curr[-1] > 1:
            groups.append(curr)
            curr = [r]
        else:
            curr.append(r)
    groups.append(curr)
    
    t_rows = groups[0]
    t_h = len(t_rows)
    t_w = inp.shape[1]
    tpl = inp[t_rows[0]:t_rows[0] + t_h, :]
    
    for grp in groups[1:]:
        grp_max_c = max(c for r in grp for c in range(t_w) if inp[r, c] == 8)
        start_r = grp[0]
        for roff in range(t_h):
            r = start_r + roff
            if r >= inp.shape[0]:
                break
            for c in range(grp_max_c + 1, t_w):
                if tpl[roff, c] == 8:
                    out[r, c] = 1
    return out

# 2. d06dbe63
def solve_d06dbe63(inp):
    out = inp.copy()
    H, W = inp.shape
    eights = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 8]
    if not eights:
        return out
    sr, sc = eights[0]
    
    # up-right
    curr_r, curr_c = sr, sc
    while True:
        nr = curr_r - 1
        if nr < 0:
            break
        out[nr, curr_c] = 5
        nr2 = nr - 1
        if nr2 < 0:
            break
        c_end = min(W, curr_c + 3)
        out[nr2, curr_c:c_end] = 5
        curr_r = nr2
        curr_c = curr_c + 2
        if curr_c >= W:
            break
            
    # down-left
    curr_r, curr_c = sr, sc
    while True:
        nr = curr_r + 1
        if nr >= H:
            break
        out[nr, curr_c] = 5
        nr2 = nr + 1
        if nr2 >= H:
            break
        c_start = max(0, curr_c - 2)
        out[nr2, c_start:curr_c + 1] = 5
        curr_r = nr2
        curr_c = curr_c - 2
        if curr_c < 0:
            break
    return out

# 3. ef135b50
def solve_ef135b50(inp):
    out = inp.copy()
    H, W = inp.shape
    lab, num = label(inp == 2)
    comps = []
    for f in range(1, num + 1):
        pts = [(r, c) for r in range(H) for c in range(W) if lab[r, c] == f]
        rmin = min(r for r, c in pts)
        rmax = max(r for r, c in pts)
        cmin = min(c for r, c in pts)
        cmax = max(c for r, c in pts)
        comps.append({
            'pts': pts,
            'rmin': rmin, 'rmax': rmax,
            'cmin': cmin, 'cmax': cmax,
            'rows': {r: [c for rr, c in pts if rr == r] for r in range(rmin, rmax + 1)}
        })
        
    for j, cj in enumerate(comps):
        best_p = None
        best_dist = 999
        for i, ci in enumerate(comps):
            if i == j: continue
            if ci['rmin'] < cj['rmin'] <= ci['rmax']:
                dist = abs(cj['cmin'] - ci['cmax'])
                if dist < best_dist:
                    best_dist = dist
                    best_p = ci
        if best_p is not None:
            ci = best_p
            start_r = cj['rmin']
            end_r = min(ci['rmax'], cj['rmax'])
            for r in range(start_r, end_r + 1):
                if r in ci['rows'] and r in cj['rows']:
                    row_ci = ci['rows'][r]
                    row_cj = cj['rows'][r]
                    left_edge = min(max(row_ci), max(row_cj))
                    right_edge = max(min(row_ci), min(row_cj))
                    for col in range(left_edge + 1, right_edge):
                        out[r, col] = 9
    return out

# 4. ea32f347
def solve_ea32f347(inp):
    out = inp.copy()
    H, W = inp.shape
    lab, num = label(inp == 5)
    comps = []
    for f in range(1, num + 1):
        pts = [(r, c) for r in range(H) for c in range(W) if lab[r, c] == f]
        comps.append((len(pts), pts))
    comps.sort(key=lambda x: x[0], reverse=True)
    rank_colors = [1, 4, 2]
    for idx, (sz, pts) in enumerate(comps):
        color = rank_colors[idx] if idx < len(rank_colors) else 2
        for r, c in pts:
            out[r, c] = color
    return out

# 5. f25fbde4
def solve_f25fbde4(inp):
    rows = [r for r in range(inp.shape[0]) if np.any(inp[r, :] != 0)]
    cols = [c for c in range(inp.shape[1]) if np.any(inp[:, c] != 0)]
    if not rows or not cols:
        return inp
    crop = inp[min(rows):max(rows) + 1, min(cols):max(cols) + 1]
    return np.kron(crop, np.ones((2, 2), dtype=int))

# 6. e98196ab
def solve_e98196ab(inp):
    div_rows = [r for r in range(inp.shape[0]) if np.all(inp[r, :] == 5)]
    if not div_rows:
        return inp
    dr = div_rows[0]
    top = inp[:dr, :]
    bot = inp[dr + 1:, :]
    return np.where(top != 0, top, bot)

# 7. a78176bb
def solve_a78176bb(inp):
    H, W = inp.shape
    out = inp.copy()
    fives = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 5]
    if not fives:
        return out
    other_colors = [c for c in np.unique(inp) if c != 0 and c != 5]
    if not other_colors:
        return out
    c_main = other_colors[0]
    
    main_pts = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == c_main]
    d0 = main_pts[0][0] - main_pts[0][1]
    
    five_diags = set(r - c for r, c in fives)
    left_count = sum(1 for d in five_diags if d < d0)
    right_count = sum(1 for d in five_diags if d > d0)
    
    out[out == 5] = 0
    if left_count > 0:
        d_target = d0 - (left_count + 2)
        for r in range(H):
            c = r - d_target
            if 0 <= c < W:
                out[r, c] = c_main
    if right_count > 0:
        d_target = d0 + (right_count + 2)
        for r in range(H):
            c = r - d_target
            if 0 <= c < W:
                out[r, c] = c_main
    return out

# 8. fcc82909
def solve_fcc82909(inp):
    out = inp.copy()
    H, W = inp.shape
    r = 0
    while r < H - 1:
        c = 0
        while c < W - 1:
            blk = inp[r:r+2, c:c+2]
            if np.all(blk != 0):
                n_cols = len(np.unique(blk))
                for dr in range(2, 2 + n_cols):
                    if r + dr < H:
                        out[r + dr, c] = 3
                        out[r + dr, c + 1] = 3
                c += 2
            else:
                c += 1
        r += 1
    return out

# 9. 539a4f51
def solve_539a4f51(inp):
    rows = [r for r in range(inp.shape[0]) if np.any(inp[r, :] != 0)]
    cols = [c for c in range(inp.shape[1]) if np.any(inp[:, c] != 0)]
    K = max(max(rows), max(cols)) + 1
    B = inp[:K, :K]
    bg = B[0, 0]
    out = np.full((10, 10), bg)
    out[:K, :K] = B
    out[:K, K:2*K] = np.tile(B[0, :], (K, 1))
    out[K:2*K, :K] = np.tile(B[:, 0][:, None], (1, K))
    out[K:2*K, K:2*K] = B
    return out

# 10. d687bc17
def solve_d687bc17(inp):
    H, W = inp.shape
    out = np.zeros_like(inp)
    out[0, :] = inp[0, :]
    out[-1, :] = inp[-1, :]
    out[:, 0] = inp[:, 0]
    out[:, -1] = inp[:, -1]
    
    top_c = inp[0, 1]
    bot_c = inp[-1, 1]
    left_c = inp[1, 0]
    right_c = inp[1, -1]
    
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            val = inp[r, c]
            if val == 0:
                continue
            if val == top_c:
                out[1, c] = val
            elif val == bot_c:
                out[H - 2, c] = val
            elif val == left_c:
                out[r, 1] = val
            elif val == right_c:
                out[r, W - 2] = val
    return out

# 11. e21d9049
def solve_e21d9049(inp):
    H, W = inp.shape
    out = np.zeros_like(inp)
    row_counts = [np.count_nonzero(inp[r, :]) for r in range(H)]
    col_counts = [np.count_nonzero(inp[:, c]) for c in range(W)]
    cross_r = np.argmax(row_counts)
    cross_c = np.argmax(col_counts)
    
    cols = [c for c in range(W) if inp[cross_r, c] != 0]
    L_h = len(cols)
    pat_h = [inp[cross_r, c] for c in cols]
    start_c = cols[0]
    for c in range(W):
        out[cross_r, c] = pat_h[(c - start_c) % L_h]
        
    rows = [r for r in range(H) if inp[r, cross_c] != 0]
    L_v = len(rows)
    pat_v = [inp[r, cross_c] for r in rows]
    start_r = rows[0]
    for r in range(H):
        out[r, cross_c] = pat_v[(r - start_r) % L_v]
    return out

# 12. 272f95fa
def solve_272f95fa(inp):
    H, W = inp.shape
    h_lines = [r for r in range(H) if np.all(inp[r, :] == 8)]
    v_lines = [c for c in range(W) if np.all(inp[:, c] == 8)]
    r0, r1 = h_lines[0], h_lines[1]
    c0, c1 = v_lines[0], v_lines[1]
    out = inp.copy()
    out[0:r0, c0+1:c1] = 2
    out[r1+1:H, c0+1:c1] = 1
    out[r0+1:r1, 0:c0] = 4
    out[r0+1:r1, c0+1:c1] = 6
    out[r0+1:r1, c1+1:W] = 3
    return out

# 13. 543a7ed5
def solve_543a7ed5(inp):
    out = inp.copy()
    H, W = inp.shape
    visited = np.zeros((H, W), dtype=bool)
    from collections import deque
    q = deque()
    for r in range(H):
        for c in [0, W - 1]:
            if inp[r, c] == 8 and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))
    for c in range(W):
        for r in [0, H - 1]:
            if inp[r, c] == 8 and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if inp[nr, nc] == 8 and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
    holes = (inp == 8) & (~visited)
    out[holes] = 4
    struct = np.ones((3, 3), dtype=bool)
    dilated_6 = binary_dilation(inp == 6, structure=struct)
    halo = dilated_6 & (inp == 8) & visited
    out[halo] = 3
    return out

# 14. 928ad970
def solve_928ad970(inp):
    out = inp.copy()
    H, W = inp.shape
    fives = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 5]
    top_r = min(r for r, c in fives)
    bot_r = max(r for r, c in fives)
    left_c = min(c for r, c in fives)
    right_c = max(c for r, c in fives)
    colors = [c for c in np.unique(inp) if c != 0 and c != 5]
    C = colors[0]
    out[top_r + 1, left_c + 1:right_c] = C
    out[bot_r - 1, left_c + 1:right_c] = C
    out[top_r + 1:bot_r, left_c + 1] = C
    out[top_r + 1:bot_r, right_c - 1] = C
    for r, c in fives:
        out[r, c] = 5
    return out

# 15. e8593010
def solve_e8593010(inp):
    out = inp.copy()
    structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    labeled, num_features = label(inp == 0, structure=structure)
    size_to_color = {1: 3, 2: 2, 3: 1}
    for feat_id in range(1, num_features + 1):
        mask = (labeled == feat_id)
        sz = np.sum(mask)
        color = size_to_color.get(sz, 0)
        out[mask] = color
    return out

# 16. 6cdd2623
def solve_6cdd2623(inp):
    H, W = inp.shape
    out = np.zeros_like(inp)
    color_lines = defaultdict(list)
    for r in range(H):
        if inp[r, 0] != 0 and inp[r, 0] == inp[r, W - 1]:
            color_lines[inp[r, 0]].append(('row', r))
    for c in range(W):
        if inp[0, c] != 0 and inp[0, c] == inp[H - 1, c]:
            color_lines[inp[0, c]].append(('col', c))
    best_color = max(color_lines.keys(), key=lambda c: len(color_lines[c]))
    for line_type, idx in color_lines[best_color]:
        if line_type == 'row':
            out[idx, :] = best_color
        else:
            out[:, idx] = best_color
    return out

# 17. b7249182
def solve_b7249182(inp):
    H, W = inp.shape
    out = np.zeros_like(inp)
    dots = [(r, c, inp[r, c]) for r in range(H) for c in range(W) if inp[r, c] != 0]
    (r1, c1, col1), (r2, c2, col2) = dots
    if c1 == c2:
        C = c1
        if r1 > r2:
            r1, r2 = r2, r1
            col1, col2 = col2, col1
        r_mid = (r1 + r2) // 2
        out[r1:r_mid, C] = col1
        out[r_mid - 1, C - 2:C + 3] = col1
        out[r_mid, C - 2] = col1
        out[r_mid, C + 2] = col1
        out[r_mid + 2:r2 + 1, C] = col2
        out[r_mid + 2, C - 2:C + 3] = col2
        out[r_mid + 1, C - 2] = col2
        out[r_mid + 1, C + 2] = col2
    elif r1 == r2:
        R = r1
        if c1 > c2:
            c1, c2 = c2, c1
            col1, col2 = col2, col1
        c_mid = (c1 + c2) // 2
        out[R, c1:c_mid] = col1
        out[R - 2:R + 3, c_mid - 1] = col1
        out[R - 2, c_mid] = col1
        out[R + 2, c_mid] = col1
        out[R, c_mid + 2:c2 + 1] = col2
        out[R - 2:R + 3, c_mid + 2] = col2
        out[R - 2, c_mid + 1] = col2
        out[R + 2, c_mid + 1] = col2
    return out

# 18. eb281b96
def solve_eb281b96(inp):
    H, W = inp.shape
    cycle = list(range(H)) + list(range(H - 2, 0, -1))
    indices = cycle + cycle + [0]
    return inp[indices, :]

# 19. fcb5c309
def solve_fcb5c309(inp):
    colors = [c for c in np.unique(inp) if c != 0]
    max_sizes = {}
    for c in colors:
        lab, num = label(inp == c)
        max_sizes[c] = max([np.sum(lab == i) for i in range(1, num + 1)])
    box_col = max(colors, key=lambda c: max_sizes[c])
    dot_col = [c for c in colors if c != box_col][0]
    lab, num = label(inp == box_col)
    best_box = None
    max_dots_inside = -1
    max_area = -1
    for feat in range(1, num + 1):
        pts = [(r, c) for r in range(inp.shape[0]) for c in range(inp.shape[1]) if lab[r, c] == feat]
        r0, r1 = min(r for r, c in pts), max(r for r, c in pts)
        c0, c1 = min(c for r, c in pts), max(c for r, c in pts)
        dots_inside = sum(1 for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if inp[r, c] == dot_col)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if dots_inside > max_dots_inside or (dots_inside == max_dots_inside and area > max_area):
            max_dots_inside = dots_inside
            max_area = area
            best_box = (r0, r1, c0, c1)
    r0, r1, c0, c1 = best_box
    crop = inp[r0:r1+1, c0:c1+1]
    return np.where((crop == box_col) | (crop == dot_col), dot_col, 0)

# 20. f15e1fac
def solve_f15e1fac(inp):
    H, W = inp.shape
    out = np.zeros_like(inp)
    eights = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 8]
    twos = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == 2]
    for r, c in twos:
        out[r, c] = 2
    if all(r == 0 for r, c in eights):
        two_cols = [c for r, c in twos]
        shift_dir = +1 if 0 in two_cols else -1
        two_rows = set(r for r, c in twos)
        current_cols = [c for r, c in eights]
        for r in range(H):
            if r in two_rows:
                current_cols = [c + shift_dir for c in current_cols]
            for c in current_cols:
                if 0 <= c < W:
                    out[r, c] = 8
    elif all(c == 0 for r, c in eights):
        two_rows = [r for r, c in twos]
        shift_dir = +1 if 0 in two_rows else -1
        two_cols = set(c for r, c in twos)
        current_rows = [r for r, c in eights]
        for c in range(W):
            if c in two_cols:
                current_rows = [r + shift_dir for r in current_rows]
            for r in current_rows:
                if 0 <= r < H:
                    out[r, c] = 8
    elif all(c == W - 1 for r, c in eights):
        two_rows = [r for r, c in twos]
        shift_dir = +1 if 0 in two_rows else -1
        two_cols = set(c for r, c in twos)
        current_rows = [r for r, c in eights]
        for c in range(W - 1, -1, -1):
            if c in two_cols:
                current_rows = [r + shift_dir for r in current_rows]
            for r in current_rows:
                if 0 <= r < H:
                    out[r, c] = 8
    for r, c in twos:
        out[r, c] = 2
    return out

SOLVERS = {
    "90f3ed37": solve_90f3ed37,
    "d06dbe63": solve_d06dbe63,
    "ef135b50": solve_ef135b50,
    "ea32f347": solve_ea32f347,
    "f25fbde4": solve_f25fbde4,
    "e98196ab": solve_e98196ab,
    "a78176bb": solve_a78176bb,
    "fcc82909": solve_fcc82909,
    "539a4f51": solve_539a4f51,
    "d687bc17": solve_d687bc17,
    "e21d9049": solve_e21d9049,
    "272f95fa": solve_272f95fa,
    "543a7ed5": solve_543a7ed5,
    "928ad970": solve_928ad970,
    "e8593010": solve_e8593010,
    "6cdd2623": solve_6cdd2623,
    "b7249182": solve_b7249182,
    "eb281b96": solve_eb281b96,
    "fcb5c309": solve_fcb5c309,
    "f15e1fac": solve_f15e1fac,
}

if __name__ == "__main__":
    passed = 0
    for tid, fn in SOLVERS.items():
        with open(f"training/{tid}.json") as f:
            d = json.load(f)
        all_ok = True
        for i, p in enumerate(d["train"]):
            pred = fn(np.array(p["input"]))
            if not np.array_equal(pred, np.array(p["output"])):
                all_ok = False
                print(f"❌ {tid} failed on train {i}!")
                break
        for i, p in enumerate(d["test"]):
            pred = fn(np.array(p["input"]))
            if not np.array_equal(pred, np.array(p["output"])):
                all_ok = False
                print(f"❌ {tid} failed on test {i}!")
                break
        if all_ok:
            passed += 1
            print(f"✅ {tid} passed 100% on all train and test!")
    print(f"\nTOTAL PASSED: {passed}/{len(SOLVERS)}")
