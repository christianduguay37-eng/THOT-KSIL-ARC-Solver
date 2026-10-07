import json
import numpy as np
from scipy.ndimage import label

def solve_264363fd(inp):
    counts = {c: int((inp==c).sum()) for c in np.unique(inp)}
    bg, fg = sorted(counts, key=counts.get, reverse=True)[:2]
    
    minor_colors = [c for c in np.unique(inp) if c not in (bg, fg)]
    minor_mask = np.isin(inp, minor_colors)
    lbl, num = label(minor_mask)
    
    template_pts = None
    seed_pts = []
    for i in range(1, num+1):
        pts = np.argwhere(lbl == i)
        if len(pts) > 1:
            template_pts = pts
        else:
            seed_pts.append(tuple(pts[0]))
            
    if template_pts is None or len(template_pts) < 3 or len(seed_pts) == 0:
        return None
        
    cr = int(round(np.mean(template_pts[:, 0])))
    cc = int(round(np.mean(template_pts[:, 1])))
    seed_color = inp[cr, cc]
    
    has_v_ray = False
    v_ray_color = None
    if (cr >= 2 and cr + 2 < inp.shape[0] and
        inp[cr-2, cc] != bg and inp[cr-1, cc] != bg and
        inp[cr+1, cc] != bg and inp[cr+2, cc] != bg and
        inp[cr-2, cc] == inp[cr-1, cc] == inp[cr+1, cc] == inp[cr+2, cc]):
        has_v_ray = True
        v_ray_color = inp[cr-1, cc]
        
    has_h_ray = False
    h_ray_color = None
    if (cc >= 2 and cc + 2 < inp.shape[1] and
        inp[cr, cc-2] != bg and inp[cr, cc-1] != bg and
        inp[cr, cc+1] != bg and inp[cr, cc+2] != bg and
        inp[cr, cc-2] == inp[cr, cc-1] == inp[cr, cc+1] == inp[cr, cc+2]):
        has_h_ray = True
        h_ray_color = inp[cr, cc-1]
        
    local_offsets = []
    for r, c in template_pts:
        if (r, c) == (cr, cc): continue
        if has_v_ray and c == cc: continue
        if has_h_ray and r == cr: continue
        local_offsets.append((r - cr, c - cc, inp[r, c]))
        
    out = inp.copy()
    for r, c in template_pts:
        out[r, c] = bg
        
    seeds = [pt for pt in seed_pts if inp[pt[0], pt[1]] == seed_color]
    H, W = inp.shape
    for sr, sc in seeds:
        if has_v_ray:
            curr_r = sr - 1
            while curr_r >= 0 and inp[curr_r, sc] == fg:
                out[curr_r, sc] = v_ray_color
                curr_r -= 1
            curr_r = sr + 1
            while curr_r < H and inp[curr_r, sc] == fg:
                out[curr_r, sc] = v_ray_color
                curr_r += 1
                
        if has_h_ray:
            curr_c = sc - 1
            while curr_c >= 0 and inp[sr, curr_c] == fg:
                out[sr, curr_c] = h_ray_color
                curr_c -= 1
            curr_c = sc + 1
            while curr_c < W and inp[sr, curr_c] == fg:
                out[sr, curr_c] = h_ray_color
                curr_c += 1
                
        for dr, dc, col in local_offsets:
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < H and 0 <= nc < W:
                if inp[nr, nc] == fg:
                    out[nr, nc] = col
                    
    return out

if __name__ == "__main__":
    with open("training/264363fd.json") as f:
        d = json.load(f)
    for i, p in enumerate(d["train"]):
        assert np.array_equal(solve_264363fd(np.array(p["input"])), np.array(p["output"]))
    for i, p in enumerate(d["test"]):
        assert np.array_equal(solve_264363fd(np.array(p["input"])), np.array(p["output"]))
    print("264363fd: 100% PASS ON ALL TRAIN AND TEST!")
