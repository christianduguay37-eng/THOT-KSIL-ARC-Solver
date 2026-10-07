import json
import numpy as np

def solve_98cf29f8(inp):
    inp = np.array(inp)
    colors = [c for c in np.unique(inp) if c != 0]
    if len(colors) != 2:
        return inp.copy()
    c1, c2 = colors
    
    def is_solid_rect(mask):
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return False
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)
        
    if is_solid_rect(inp == c1) and not is_solid_rect(inp == c2):
        c_fix, c_mob = c1, c2
    elif is_solid_rect(inp == c2) and not is_solid_rect(inp == c1):
        c_fix, c_mob = c2, c1
    else:
        coords1 = np.argwhere(inp == c1)
        coords2 = np.argwhere(inp == c2)
        dens1 = len(coords1) / ((coords1.max(0) - coords1.min(0) + 1).prod())
        dens2 = len(coords2) / ((coords2.max(0) - coords2.min(0) + 1).prod())
        if dens1 > dens2:
            c_fix, c_mob = c1, c2
        else:
            c_fix, c_mob = c2, c1
            
    mob_mask = (inp == c_mob)
    fix_mask = (inp == c_fix)
    
    adj_stem_pixels = []
    for r, c in np.argwhere(mob_mask):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1] and fix_mask[nr, nc]:
                adj_stem_pixels.append(((r, c), (-dr, -dc)))
                
    if not adj_stem_pixels:
        return inp.copy()
        
    start_p, stem_dir = adj_stem_pixels[0]
    dr, dc = stem_dir
    
    stem_pixels = []
    curr = start_p
    while True:
        r, c = curr
        if not (0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]) or not mob_mask[r, c]:
            break
        perp_dr, perp_dc = dc, dr
        p1 = (r + perp_dr, c + perp_dc)
        p2 = (r - perp_dr, c - perp_dc)
        has_p1 = (0 <= p1[0] < inp.shape[0] and 0 <= p1[1] < inp.shape[1] and mob_mask[p1[0], p1[1]])
        has_p2 = (0 <= p2[0] < inp.shape[0] and 0 <= p2[1] < inp.shape[1] and mob_mask[p2[0], p2[1]])
        if has_p1 or has_p2:
            break
        stem_pixels.append((r, c))
        curr = (r + dr, c + dc)
        
    stem_len = len(stem_pixels)
    shift_r = -dr * stem_len
    shift_c = -dc * stem_len
    
    block_pixels = [p for p in np.argwhere(mob_mask) if tuple(p) not in set(stem_pixels)]
    
    out = inp.copy()
    out[mob_mask] = 0
    for r, c in block_pixels:
        nr, nc = r + shift_r, c + shift_c
        out[nr, nc] = c_mob
        
    return out

if __name__ == "__main__":
    with open("training/98cf29f8.json") as f:
        d = json.load(f)
    
    all_ok = True
    for i, ex in enumerate(d["train"]):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pred = solve_98cf29f8(inp)
        if not np.array_equal(pred, out):
            print(f"FAILED Train {i}")
            all_ok = False
        else:
            print(f"PASS Train {i}")
            
    for i, ex in enumerate(d["test"]):
        inp = np.array(ex["input"])
        pred = solve_98cf29f8(inp)
        print(f"Test {i} output shape: {pred.shape}")
        
    if all_ok:
        print("ALL TESTS PASSED for 98cf29f8!")
