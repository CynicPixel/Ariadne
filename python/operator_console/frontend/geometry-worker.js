// Decode the standard 32-byte splat records off the UI thread. Sampling only
// affects display density; reconstruction artifacts retain every observed point.
self.onmessage = ({data:{id,bytes,limit}}) => {
  const view = new DataView(bytes), n = Math.floor(bytes.byteLength/32);
  const step = Math.max(1,Math.ceil(n/limit)), count = Math.ceil(n/step);
  const xyz = new Float32Array(count*3), rgb = new Float32Array(count*3);
  let j=0;
  for(let i=0;i<n;i+=step){
    const at=i*32;
    const p=[0,4,8].map(o=>view.getFloat32(at+o,true));
    if(!p.every(Number.isFinite))continue;
    for(let k=0;k<3;k++){xyz[j*3+k]=p[k];rgb[j*3+k]=view.getUint8(at+24+k)/255;}
    j++;
  }
  self.postMessage({id,xyz,rgb,count:j},[xyz.buffer,rgb.buffer]);
};
