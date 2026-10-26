"""Prepare textured camera views, then optimize a Gaussian model with CUDA gsplat.

Source is a synthetic WorldGen scene. This is not a robot mission measurement.
CPU prepare runs in the Manim image; train runs in the existing reconstruction GPU image.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image

DT=np.dtype([('p','<f4',3),('s','<f4',3),('c','u1',4),('q','u1',4)])


def prepare(scene: Path, seed_splat: Path, out: Path):
    out.mkdir(parents=True,exist_ok=True)
    scene_json=json.loads((scene/'scene.json').read_text())
    surfaces=[]
    for mesh in sorted((scene/'meshes').glob('*.obj')):
        if mesh.stem.startswith('ceiling_'): continue
        v=np.array([[float(x) for x in line.split()[1:]] for line in mesh.read_text().splitlines() if line.startswith('v ')])
        eu,ev=v[1]-v[0],v[3]-v[0]
        n=np.cross(eu,ev);n/=np.linalg.norm(n)
        with Image.open(scene/'textures'/f'{mesh.stem}.png') as im:
            tex=np.array(im.convert('RGB'),dtype=np.float32)/255
        tex*=0.72+0.28*abs(float(n@[0.3,-0.4,0.866]))
        surfaces.append((v[0],eu,ev,n,tex))
    extent=np.concatenate([np.array([o,o+eu+ev]) for o,eu,ev,n,tex in surfaces])
    centre=(extent.min(0)+extent.max(0))/2;centre[2]=0.8
    radius=max(np.ptp(extent[:,0]),np.ptp(extent[:,1]))*.9
    views=[]
    for elev in [28,50,70]:
        for az in np.linspace(0,2*np.pi,12,endpoint=False):
            el=np.deg2rad(elev)
            pos=centre+radius*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
            views.append((pos,centre))
    rooms=scene_json['topology']['rooms']
    for rect in rooms.values():
        x0,y0,x1,y1=rect
        pos=np.array([(x0+x1)/2,(y0+y1)/2,1.4])
        for angle in np.linspace(0,2*np.pi,8,endpoint=False):
            views.append((pos,pos+np.array([np.cos(angle),np.sin(angle),-0.08])))
    w,h=320,180
    fy=h/(2*np.tan(np.deg2rad(65)/2))
    K=np.array([[fy,0,w/2],[0,fy,h/2],[0,0,1]],dtype=np.float32)
    yy,xx=np.mgrid[:h,:w]
    ray_cam=np.stack(((xx.ravel()+.5-w/2)/fy,(yy.ravel()+.5-h/2)/fy,np.ones(w*h)),axis=1)
    images,depths,matrices=[],[],[]
    for idx,(pos,target) in enumerate(views):
        f=(target-pos)/np.linalg.norm(target-pos)
        right=np.cross(f,[0,0,1]);right/=np.linalg.norm(right)
        down=np.cross(f,right);R=np.stack((right,down,f))
        rays=ray_cam@R
        depth=np.full(w*h,np.inf);rgb=np.zeros((w*h,3),dtype=np.float32)
        for origin,eu,ev,normal,tex in surfaces:
            denom=rays[:,0]*normal[0]+rays[:,1]*normal[1]+rays[:,2]*normal[2]
            t=np.divide(float((origin-pos)@normal),denom,out=np.full_like(denom,np.inf),where=np.abs(denom)>1e-8)
            offset=pos-origin
            ru=rays[:,0]*eu[0]+rays[:,1]*eu[1]+rays[:,2]*eu[2]
            rv=rays[:,0]*ev[0]+rays[:,1]*ev[1]+rays[:,2]*ev[2]
            with np.errstate(invalid='ignore'):
                u=(float(offset@eu)+t*ru)/float(eu@eu)
                v=(float(offset@ev)+t*rv)/float(ev@ev)
            use=(t>.05)&(t<depth)&(u>=0)&(u<=1)&(v>=0)&(v<=1)
            ids=np.flatnonzero(use)
            if len(ids):
                th,tw=tex.shape[:2]
                rgb[ids]=tex[np.minimum(th-1,((1-v[ids])*th).astype(int)),np.minimum(tw-1,(u[ids]*tw).astype(int))]
                depth[ids]=t[ids]
        vm=np.eye(4,dtype=np.float32);vm[:3,:3]=R;vm[:3,3]=-R@pos
        image=np.round(rgb.reshape(h,w,3)*255).astype(np.uint8)
        images.append(image);depths.append(np.nan_to_num(depth.reshape(h,w),posinf=0).astype(np.float32));matrices.append(vm)
        if idx%12==0: print('prepared views',idx,len(views),flush=True)
    np.savez_compressed(out/'views.npz',images=np.array(images),depth=np.array(depths),viewmats=np.array(matrices),K=K)
    (out/'seed.splat').write_bytes(seed_splat.read_bytes())
    (out/'data_manifest.json').write_text(json.dumps({'source':'WorldGen textured camera renders','scene_id':scene_json['scene_id'],'views':len(views),'size':[w,h],'seed_sha256':hashlib.sha256(seed_splat.read_bytes()).hexdigest(),'roof':'omitted'},indent=2))
    Image.fromarray(images[3]).save(out/'example_view.png')
    print('prepared',len(views),'camera views',flush=True)


def train(data: Path,out: Path,steps: int):
    import torch
    import gsplat
    from gsplat import rasterization
    torch.set_num_threads(2);torch.manual_seed(626)
    dev='cuda';out.mkdir(parents=True,exist_ok=True)
    dataset=np.load(data/'views.npz');seed=np.fromfile(data/'seed.splat',dtype=DT)
    T=lambda x:torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32,device=dev)
    params={
      'means':torch.nn.Parameter(T(seed['p'])),
      'scales':torch.nn.Parameter(torch.log(T(seed['s']).clamp_min(.001))),
      'quats':torch.nn.Parameter(torch.nn.functional.normalize(T((seed['q'].astype(float)-128)/128),dim=-1)),
      'opacity':torch.nn.Parameter(torch.logit(T(seed['c'][:,3]/255).clamp(.01,.99))),
      'colour':torch.nn.Parameter(torch.logit(T(seed['c'][:,:3]/255).clamp(.01,.99)))}
    rates={'means':1e-4,'scales':2e-3,'quats':5e-4,'opacity':1e-2,'colour':1e-2}
    opt=torch.optim.Adam([{'params':[p],'lr':rates[k]} for k,p in params.items()],eps=1e-15)
    images=T(dataset['images']/255);views=T(dataset['viewmats']);K=T(dataset['K'])[None];depths=T(dataset['depth'])
    nv,h,w,_=images.shape
    train_ids=[i for i in range(nv) if i%12!=5];holdout=[i for i in range(nv) if i%12==5]
    def render(i):
        image,alpha,_=rasterization(params['means'],torch.nn.functional.normalize(params['quats'],dim=-1),params['scales'].exp(),params['opacity'].sigmoid(),params['colour'].sigmoid(),views[i:i+1],K,w,h,packed=False,render_mode='RGB+ED',near_plane=.05,far_plane=100.)
        return image[0],alpha[0,...,0]
    @torch.no_grad()
    def score(ids):
        ms=[]
        for i in ids:
            im,_=render(i);ms.append(float((im[...,:3]-images[i]).square().mean()))
        return -10*np.log10(max(np.mean(ms),1e-12))
    before=score(holdout);rng=np.random.default_rng(626);start=time.time();log=[]
    for step in range(steps):
        i=int(rng.choice(train_ids));im,alpha=render(i)
        colour=(im[...,:3]-images[i]).abs().mean()
        valid=depths[i]>.05
        depth=(im[...,3][valid]-depths[i][valid]).abs().clamp(max=1).mean()
        # RGB fitting drives appearance; a small depth term keeps learned geometry on the rendered surfaces.
        loss=colour+.02*depth
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        with torch.no_grad():
            params['scales'].clamp_(np.log(.001),np.log(.3));params['opacity'].clamp_(-5,8)
        if step%100==0 or step==steps-1:
            rec={'step':step,'loss':float(loss),'rgb_l1':float(colour),'depth_l1':float(depth),'elapsed_s':time.time()-start}
            log.append(rec);print(json.dumps(rec),flush=True)
    after=score(holdout)
    result=np.empty(len(seed),dtype=DT)
    result['p']=params['means'].detach().cpu().numpy();result['s']=params['scales'].exp().detach().cpu().numpy()
    result['q']=np.clip(np.round(torch.nn.functional.normalize(params['quats'],dim=-1).detach().cpu().numpy()*128+128),0,255).astype(np.uint8)
    result['c']=np.round(torch.cat((params['colour'].sigmoid(),params['opacity'].sigmoid()[:,None]),dim=1).detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
    result.tofile(out/'trained_worldgen.splat')
    for i in holdout[:3]:
        with torch.no_grad():pred,_=render(i)
        Image.fromarray(np.round(pred[...,:3].clamp(0,1).cpu().numpy()*255).astype(np.uint8)).save(out/f'heldout_{i:03d}.png')
        Image.fromarray(dataset['images'][i]).save(out/f'target_{i:03d}.png')
    manifest={'source':'WorldGen rendered camera views','measured_robot_run':False,'training':'CUDA gsplat gradient optimization','gsplat':gsplat.__version__,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'steps':steps,'gaussians':len(result),'train_views':len(train_ids),'heldout_views':len(holdout),'heldout_psnr_before':before,'heldout_psnr_after':after,'wall_s':time.time()-start,'sha256':hashlib.sha256((out/'trained_worldgen.splat').read_bytes()).hexdigest(),'data':json.loads((data/'data_manifest.json').read_text()),'log':log}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare');p.add_argument('--scene',type=Path,required=True);p.add_argument('--seed',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('train');p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=3000)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.scene,args.seed,args.out)
    else:train(args.data,args.out,args.steps)
