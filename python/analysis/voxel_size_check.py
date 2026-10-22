"""Usage: python3 -I voxel_size_check.py SCENES_DIR   (SCENES_DIR holds ci_tiny_0/ and maze_medium_0/ with scene_gt.npz)

D110 voxel-size check, redone with the GT's own clearance convention (worldgen raster._edt_to: centre to the
nearest point of a blocked square). Same conservative rasterisation and one-cell bleed as the first check."""
import sys, math
from collections import deque
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dataset_tools'))
from worldgen.raster import _edt_to

def dilate(b):
    o=b.copy(); o[1:]|=b[:-1]; o[:-1]|=b[1:]; o[:,1:]|=b[:,:-1]; o[:,:-1]|=b[:,1:]
    o[1:,1:]|=b[:-1,:-1]; o[:-1,:-1]|=b[1:,1:]; o[1:,:-1]|=b[:-1,1:]; o[:-1,1:]|=b[1:,:-1]; return o
def conn(trav,a,t):
    if not (trav[a] and trav[t]): return False
    seen=np.zeros_like(trav); dq=deque([a]); seen[a]=True
    while dq:
        y,x=dq.popleft()
        if (y,x)==t: return True
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                yy,xx=y+dy,x+dx
                if 0<=yy<trav.shape[0] and 0<=xx<trav.shape[1] and trav[yy,xx] and not seen[yy,xx]:
                    seen[yy,xx]=True; dq.append((yy,xx))
    return False
for s in ('ci_tiny_0','maze_medium_0'):
    z=np.load(Path(sys.argv[1]) / s / 'scene_gt.npz')
    blocked=~z['free']; g=float(z['grid_m']); H,W=blocked.shape
    # sanity: at the GT resolution itself, the convention must reproduce GT trav_responder exactly
    gt_tr = (_edt_to(blocked, 20) * g >= 0.40 - 1e-9) & ~blocked
    print(f'{s}: reproduces GT trav_responder at 5 cm with no bleed: {bool((gt_tr == z["trav_responder"]).all())}')
    for vox in (0.05,0.08,0.10):
        nh,nw=int(H*g/vox),int(W*g/vox); r=vox/g
        cb=np.zeros((nh,nw),bool)
        for dy in np.linspace(0,r-1e-6,5):
            for dx in np.linspace(0,r-1e-6,5):
                cb|=blocked[np.minimum((np.arange(nh)[:,None]*r+dy).astype(int),H-1),np.minimum((np.arange(nw)[None,:]*r+dx).astype(int),W-1)]
        cell=lambda c:(min(int((c[0]+.5)*g/vox),nh-1),min(int((c[1]+.5)*g/vox),nw-1))
        for bleed in (0,1):
            b=dilate(cb) if bleed else cb
            trav=(_edt_to(b, int(math.ceil(0.5/vox))+2)*vox >= 0.40-1e-9) & ~b
            res=[conn(trav,cell(st),cell(tuple(z['target_cell']))) for st in z['start_cells']]
            print(f'  vox {vox:.2f} bleed {bleed}: start0/start1 -> target at rho_resp: {res}', flush=True)
