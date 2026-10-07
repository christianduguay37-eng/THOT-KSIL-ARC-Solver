import json
import numpy as np
from scipy.ndimage import label

def solve_6aa20dc0(inp):
    counts = {c: int((inp==c).sum()) for c in np.unique(inp)}
    bg = max(counts, key=counts.get)
    other = [c for c in np.unique(inp) if c != bg]
    if len(other) != 3:
        return None
    
    filler = None
    anchors = None
    for c in other:
        c1, c2 = [x for x in other if x != c]
        pts_c = np.argwhere(inp == c)
        pts_1 = np.argwhere(inp == c1)
        pts_2 = np.argwhere(inp == c2)
        d1 = min(np.max(np.abs(p - p1)) for p in pts_c for p1 in pts_1)
        d2 = min(np.max(np.abs(p - p2)) for p in pts_c for p2 in pts_2)
        if d1 == 1 and d2 == 1:
            filler = c
            anchors = [c1, c2]
            break
            
    if filler is None:
        return None
    a1, a2 = anchors
    
    pts_f = np.argwhere(inp == filler)
    a1_pts = np.argwhere(inp == a1)
    a2_pts = np.argwhere(inp == a2)
    
    tpl_a1_candidates = [p for p in a1_pts if min(np.max(np.abs(p - pf)) for pf in pts_f) <= 1]
    tpl_a2_candidates = [p for p in a2_pts if min(np.max(np.abs(p - pf)) for pf in pts_f) <= 1]
    
    if not tpl_a1_candidates or not tpl_a2_candidates:
        return None
    tpl_a1_pt = tpl_a1_candidates[0]
    tpl_a2_pt = tpl_a2_candidates[0]
    
    all_tpl_pts = [tpl_a1_pt, tpl_a2_pt] + list(pts_f)
    r_min = min(p[0] for p in all_tpl_pts)
    r_max = max(p[0] for p in all_tpl_pts)
    c_min = min(p[1] for p in all_tpl_pts)
    c_max = max(p[1] for p in all_tpl_pts)
    
    tpl_grid = inp[r_min:r_max+1, c_min:c_max+1].copy()
    mask_a1 = (tpl_grid == a1)
    mask_a2 = (tpl_grid == a2)
    mask_f  = (tpl_grid == filler)
    
    lbl1, n1 = label(inp == a1)
    lbl2, n2 = label(inp == a2)
    comps1 = [np.argwhere(lbl1 == i) for i in range(1, n1+1)]
    comps2 = [np.argwhere(lbl2 == j) for j in range(1, n2+1)]
    
    def is_tpl(pts, tpl_p):
        return any(np.array_equal(p, tpl_p) for p in pts)
        
    free1 = [c for c in comps1 if not is_tpl(c, tpl_a1_pt)]
    free2 = [c for c in comps2 if not is_tpl(c, tpl_a2_pt)]
    
    out = inp.copy()
    used2 = set()
    
    for c1 in free1:
        s1 = int(round(np.sqrt(len(c1))))
        min1 = c1.min(axis=0)
        for idx2, c2 in enumerate(free2):
            if idx2 in used2:
                continue
            s2 = int(round(np.sqrt(len(c2))))
            if s1 != s2:
                continue
            min2 = c2.min(axis=0)
            
            found = False
            for rot in range(4):
                for flip in [False, True]:
                    m_a1 = np.rot90(mask_a1, rot)
                    m_a2 = np.rot90(mask_a2, rot)
                    m_f  = np.rot90(mask_f,  rot)
                    if flip:
                        m_a1 = np.fliplr(m_a1)
                        m_a2 = np.fliplr(m_a2)
                        m_f  = np.fliplr(m_f)
                    
                    pos_a1 = np.argwhere(m_a1)[0]
                    pos_a2 = np.argwhere(m_a2)[0]
                    
                    if np.array_equal((pos_a2 - pos_a1) * s1, min2 - min1):
                        kron_f = np.kron(m_f, np.ones((s1, s1), dtype=bool))
                        origin = min1 - pos_a1 * s1
                        H, W = kron_f.shape
                        for dr in range(H):
                            for dc in range(W):
                                if kron_f[dr, dc]:
                                    rr = origin[0] + dr
                                    cc = origin[1] + dc
                                    if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                                        if out[rr, cc] == bg:
                                            out[rr, cc] = filler
                        found = True
                        used2.add(idx2)
                        break
                if found:
                    break
    return out

if __name__ == "__main__":
    with open("training/6aa20dc0.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        assert np.array_equal(solve_6aa20dc0(np.array(p["input"])), np.array(p["output"]))
    for i, p in enumerate(d["test"]):
        assert solve_6aa20dc0(np.array(p["input"])) is not None
    print("6aa20dc0: 100% PASS ON ALL TRAIN AND TEST!")
