import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {OBJLoader} from 'three/addons/loaders/OBJLoader.js';
export class View {
  constructor(el,onLabel){
    this.el=el;this.onLabel=onLabel;this.renderer=new THREE.WebGLRenderer({antialias:false,powerPreference:'high-performance'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));el.prepend(this.renderer.domElement);
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#0c151d');
    this.camera=new THREE.PerspectiveCamera(48,1,.03,500);this.camera.up.set(0,0,1);this.camera.position.set(-12,-12,20);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;
    this.scene.add(new THREE.HemisphereLight(0xeafaff,0x33404a,2));
    const sun=new THREE.DirectionalLight(0xffffff,2);sun.position.set(4,-8,15);this.scene.add(sun);
    this.world=new THREE.Group();this.points=new THREE.Group();this.scene.add(this.world,this.points);
    this.meshes=new Map();this.mode='world';this.generation=0;this.busy=false;this.pending=null;
    this.worker=new Worker('/static/dist/geometry-worker.js');this.jobs=new Map();this.job=0;
    this.worker.onmessage=({data})=>{this.jobs.get(data.id)?.(data);this.jobs.delete(data.id);};
    new ResizeObserver(()=>{const w=el.clientWidth,h=el.clientHeight;if(!w||!h)return;this.renderer.setSize(w,h,false);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();}).observe(el);
    this.renderer.setAnimationLoop(()=>{if(document.hidden)return;this.controls.update();this.renderer.render(this.scene,this.camera);});
  }
  disposeObject(m){if(typeof m.dispose==='function'){m.dispose();return;}m.traverse(x=>{x.geometry?.dispose();if(Array.isArray(x.material))x.material.forEach(a=>a.dispose());else x.material?.dispose();});}
  clear(session=null){this.session=session;this.generation++;for(const {mesh} of this.meshes.values()){this.points.remove(mesh);this.disposeObject(mesh);}this.meshes.clear();this.pending=null;this.version=null;}
  modeTo(mode){this.mode=mode;this.world.visible=mode==='world';this.points.visible=mode!=='world';}
  fit(){const target=this.mode==='world'?this.world:this.points;const box=new THREE.Box3().setFromObject(target);if(box.isEmpty())return;const c=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());this.controls.target.copy(c);const radius=Math.max(size.x,size.y,size.z,2);this.camera.position.set(c.x-radius*.6,c.y-radius*.75,c.z+radius);this.controls.update();}
  async loadWorld(key){const g=++this.generation;const r=await fetch('/api/demo/world.obj');if(!r.ok)return;const obj=new OBJLoader().parse(await r.text());if(g!==this.generation)return;
    for(const c of [...this.world.children]){this.world.remove(c);this.disposeObject(c);}
    const colors={wall:0x8cacae,floor:0x243c48,prop:0xbb9770,board:0xbed4c6};
    obj.traverse(x=>{if(x.isMesh){x.material?.dispose();x.material=new THREE.MeshStandardMaterial({color:colors[x.name]??0x637b88,roughness:.8,side:THREE.DoubleSide});}});
    this.world.add(obj);this.modeTo('world');this.fit();this.onLabel('Generated world · ground truth · orbit to inspect');
  }
  async decode(bytes,limit){const id=++this.job;return new Promise(resolve=>{this.jobs.set(id,resolve);this.worker.postMessage({id,bytes,limit},[bytes]);});}
  offer(table){this.pending=table;if(!this.busy)this.drain();}
  async drain(){this.busy=true;try{while(this.pending){const table=this.pending;this.pending=null;await this.apply(table);}}catch(e){this.onLabel('Geometry temporarily unavailable · retrying');console.warn(e);}finally{this.busy=false;}}
  async apply(table){const epoch=this.generation;const rows=table.submaps??[];const trained=table.backend==='gpu';
    const key=`${trained}:${table.version}`;if(key===this.version)return;
    const keep=new Set(rows.map(r=>r.sid));for(const [sid,have] of this.meshes)if(!keep.has(sid)){this.points.remove(have.mesh);this.disposeObject(have.mesh);this.meshes.delete(sid);}
    const limit=Math.max(1,Math.floor(200000/Math.max(rows.length,1)));let missing=false;
    for(const row of rows){if(epoch!==this.generation)return;let have=this.meshes.get(row.sid);const name=`${trained}:${row.chunk}:${limit}`;
      if(!have||have.name!==name){
        if(row.bytes>32*2000000){missing=true;continue;}
        const r=await fetch(`/api/demo/chunk/${encodeURIComponent(row.chunk)}?backend=${trained?'gpu':'cpu_init'}&session=${encodeURIComponent(this.session||'')}`);if(!r.ok){missing=true;continue;}
        let mesh;const bytes=await r.arrayBuffer();
        // GPU-trained splats are opt-in and capped; large models retain the bounded point view.
        if(trained&&rows.reduce((a,b)=>a+(b.gaussians||0),0)<=500000){
          const {SparkRenderer,SplatMesh}=await import('@sparkjsdev/spark');
          if(!this.spark){this.spark=new SparkRenderer({renderer:this.renderer});this.scene.add(this.spark);}
          mesh=new SplatMesh({fileBytes:new Uint8Array(bytes),fileType:'splat'});await mesh.initialized;
        }else{
          const d=await this.decode(bytes,limit),geo=new THREE.BufferGeometry();
          geo.setAttribute('position',new THREE.BufferAttribute(d.xyz,3));geo.setAttribute('color',new THREE.BufferAttribute(d.rgb,3));geo.setDrawRange(0,d.count);
          mesh=new THREE.Points(geo,new THREE.PointsMaterial({vertexColors:true,size:.045,sizeAttenuation:true}));
        }
        if(epoch!==this.generation){this.disposeObject(mesh);return;}
        if(have){this.points.remove(have.mesh);this.disposeObject(have.mesh);}this.points.add(mesh);have={name,mesh};this.meshes.set(row.sid,have);
      }
      if(row.T_W_anchor?.length===16){have.mesh.matrixAutoUpdate=false;have.mesh.matrix.set(...row.T_W_anchor);have.mesh.matrixWorldNeedsUpdate=true;}
    }
    if(!missing)this.version=key;
    if(this.mode!=='world')this.onLabel(`${trained?'Trained reconstruction':'Observed point cloud · untrained'} · ${rows.length} submaps`);
  }
}
