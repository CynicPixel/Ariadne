const $=id=>document.getElementById(id);
const colors=['#76e7ca','#76bcff','#efa8ef'];
let state={},frame={},token=sessionStorage.getItem('ariadne-controller')||'',controlled=false;
let sceneKey='',sessionKey='',entrances=[],selected=[0,1],target=null,valid=false,view,mode='world',setup=true;
let mapImage=null,mapImageKey='',region=[],action=null,eventIndex=0,replaying=false,playbackTimer=null;
let detailPages=[],detailIndex=0,latestAt=0,playingFrames=null,decisionKey="",validationSeq=0;
const cameraCache=new Map();
function notice(text){$('notice').textContent=text;}
async function api(action,body={},key=crypto.randomUUID()){
 const r=await fetch(`/api/demo/${action}`,{method:'POST',headers:{'Content-Type':'application/json','X-Controller-Token':token,'Idempotency-Key':key},body:JSON.stringify(body)});
 const d=await r.json();if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:JSON.stringify(d.detail));return d;
}
function on(id,fn){$(id).addEventListener('click',()=>Promise.resolve().then(fn).catch(e=>notice(e.message)));}
function time(t=0){return `${Math.floor(t/60).toString().padStart(2,'0')}:${Math.floor(t%60).toString().padStart(2,'0')}`;}
function claimLabel(){ $('claim').textContent=controlled?'You have control':state.controller_remaining_s>0?'Spectating':'Take control';}
on('claim',async()=>{const d=await api('lease');token=d.token;sessionStorage.setItem('ariadne-controller',token);controlled=true;claimLabel();notice('Controller connected. Choose a world.');updateControls();if(state.state==='idle')await generate();else if(sceneKey)await validate();});
setInterval(async()=>{if(!controlled)return;try{await api('heartbeat');}catch(e){controlled=false;claimLabel();updateControls();notice(e.message);}},5000);
async function generate(){valid=false;target=null;selected=Array.from({length:+$('agent-count').value},(_,i)=>i);await api('generate',{preset:$('preset').value,seed:+$('seed').value});setup=true;notice('Generating a reproducible scene…');}
on('generate',generate);on('random',async()=>{$('seed').value=crypto.getRandomValues(new Uint32Array(1))[0]&0x7fffffff;await generate();});
async function validate(){selected.sort((a,b)=>a-b);const seq=++validationSeq;valid=false;updateControls();if(!target||!controlled)return;try{await api('validate',{entrances:[...selected].sort((a,b)=>a-b),target});if(seq!==validationSeq)return;valid=true;notice('Mission validated. Ready to deploy.');}catch(e){if(seq!==validationSeq)return;notice(e.message);}updateControls();drawMap();}
$('agent-count').onchange=()=>{selected=Array.from({length:Math.min(+$('agent-count').value,entrances.length)},(_,i)=>i);validate();};
on('start',async()=>{await validate();if(!valid)return;await api('start',{entrances:[...selected].sort((a,b)=>a-b),target,duration_s:+$('duration').value});setup=false;mapImage=null;mapImageKey='';switchView('observed');notice('Starting simulation and sensor pipeline…');});
on('hold',async()=>{await api('hold');notice('Hold queued; waiting for runtime acknowledgment.');});
on('stop',async()=>{await api('stop');notice('Ending mission; preserving observations and decisions.');});
on('new-run',()=>{stopPlayback();setup=true;target=state.meta?.target_xy;selected=Array.from({length:Math.min(2,entrances.length)},(_,i)=>i);switchView('world');updateControls();loadMap();validate();});
on('train',async()=>{await api('train');notice('Training splats from recorded observations.');});
function updateControls(){const active=['starting','running','stopping'].includes(state.state),busy=active||state.state==='generating'||state.reconstruction==='training';
 $('setup').hidden=!setup||active;for(const id of ['generate','random','preset','seed','agent-count','duration'])$(id).disabled=!controlled||busy;
 $('start').disabled=!controlled||!valid||busy;$('hold').disabled=!controlled||state.state!=='running';$('stop').disabled=!controlled||!active;
 $('new-run').hidden=setup||active||!state.session_id;$('playback').hidden=!state.session_id||active;$('train').hidden=!state.session_id||active;
 $('train').disabled=!controlled||state.reconstruction==='training'||state.reconstruction==='trained';$('train').textContent=state.reconstruction==='training'?'Training…':'Train splats';
 document.querySelectorAll('[data-action]').forEach(b=>b.disabled=!controlled||state.state!=='running'||replaying);
 $('map-title').textContent=setup?'Mission briefing':'Observed map';$('map-mode').textContent=setup?'GROUND TRUTH':replaying?'RECORDED':'ESTIMATED';
 $('map-hint').textContent=action?'Click polygon vertices; Apply to confirm.':setup?'Select entry markers; click a free cell for the target.':'Select an agent or draw a mission region.';
}
async function switchView(next){mode=next;view?.modeTo(next);$('view-title').textContent=next==='world'?'Generated world · ground truth':'Observed reconstruction';$('view-toggle').textContent=next==='world'?'View reconstruction':'View world';$('geometry-label').textContent=next==='world'?'Generated world · not operator knowledge':'Waiting for observed geometry';if(next!=='world'&&frame.reconstruction)view?.offer(frame.reconstruction);}
on('view-toggle',()=>switchView(mode==='world'?'observed':'world'));on('reset-view',()=>view?.fit());
async function initializeView(){try{const {View}=await import('./view.js');view=new View($('viewport'),label=>$('geometry-label').textContent=label);view.clear(sessionKey||null);if(sceneKey){await view.loadWorld(sceneKey);view.modeTo(mode);}}catch(e){notice('3D renderer unavailable. Map and controls remain usable.');console.error(e);}}
initializeView();
function updateSelect(id,n){const s=$(id);if(s.options.length===n)return;const previous=s.value;s.replaceChildren(...Array.from({length:n},(_,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`Agent ${i+1}`;return o;}));s.value=+previous<n?previous:'0';}
async function receive(d){state=d;latestAt=performance.now();$('connection').textContent=d.enabled?'Worker connected':'Live worker disabled';$('status').textContent=d.state;$('clock').textContent=time(frame.t_s);claimLabel();
 if(d.error)notice(d.error);
 for(const name of ['github','paper','video']){const a=$(name),url=d.links?.[name];if(url&&(/^(https?:\/\/|\/api\/)/).test(url)){a.href=url;a.hidden=false;}else a.hidden=true;}
 if($('agent-count').options.length!==d.max_agents){$('agent-count').replaceChildren(...Array.from({length:d.max_agents},(_,i)=>new Option(String(i+1),String(i+1))));$('agent-count').value=Math.min(2,d.max_agents);}
 const key=d.scene?.scene_id?`${d.scene.scene_id}:${d.scene.preview_id||''}`:null;if(key&&key!==sceneKey&&d.meta){sceneKey=key;sessionKey='';setup=true;mapImage=null;target=d.meta.target_xy;$('seed-label').textContent=`SEED ${d.scene.seed}`;$('seed').value=d.scene.seed;$('preset').value=d.scene.preset;
 entrances=await(await fetch('/api/demo/entrances')).json();selected=Array.from({length:Math.min(+$('agent-count').value,entrances.length)},(_,i)=>i);$('view-empty').hidden=true;view?.clear();await view?.loadWorld(key);switchView('world');validate();}
 if(d.session_id&&d.session_id!==sessionKey){sessionKey=d.session_id;view?.clear(sessionKey);cameraCache.clear();mapImage=null;mapImageKey='';if(['running','starting','stopping','completed'].includes(d.state)){setup=false;switchView('observed');}}
 if(!replaying){frame=d.frame||{};renderFrame();}updateControls();loadMap();
 const commands=(d.commands||[]).filter(c=>c.kind);const last=commands.at(-1);$('command-status').textContent=last?`${last.kind.replaceAll('_',' ')} · ${last.state}${last.detail?.error?` · ${last.detail.error}`:''}`:'No pending commands';
}
function renderFrame(){const agents=frame.agents||[];updateSelect('agent',Math.max(agents.length,selected.length,1));updateSelect('camera-agent',Math.max(agents.length,selected.length,1));$('clock').textContent=time(frame.t_s);renderDecision();drawMap();refreshCameras();if(frame.reconstruction)view?.offer(frame.reconstruction);$('metric').textContent=`${((frame.bytes?.total||0)/1024).toFixed(1)} KiB exchanged · ${state.reconstruction||'waiting'}`;}
let socket,offline=false;
function connect(){socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/api/demo/stream`);let processing=false,pending,opened=false;
 socket.onopen=()=>{opened=true;offline=false;$('claim').disabled=false;};
 socket.onmessage=e=>{pending=JSON.parse(e.data);if(processing)return;processing=true;(async()=>{try{while(pending){const d=pending;pending=null;await receive(d);}}catch(e){notice(e.message);}finally{processing=false;}})();};
 socket.onclose=()=>{if(!opened&&!sessionKey&&!sceneKey){goOffline();return;}$('connection').textContent='Reconnecting…';setTimeout(connect,1500);};}
// Static hosting (no live worker behind the endpoint): one status snapshot, controls off, no reconnect loop.
async function goOffline(){try{const r=await fetch('/api/demo/status');const d=r.ok&&(r.headers.get('content-type')||'').includes('json')?await r.json():null;
 if(d?.enabled){setTimeout(connect,1500);return;}offline=true;$('claim').disabled=true;if(d)await receive(d);
 $('connection').textContent='Live worker offline · generated scene shown';notice('The live simulation worker is offline. This shows a generated scene; a live run is available on request.');}
 catch(e){offline=true;$('claim').disabled=true;$('connection').textContent='Live worker offline';}}
connect();
if(token)api('heartbeat').then(()=>{controlled=true;updateControls();claimLabel();}).catch(()=>{token='';sessionStorage.removeItem('ariadne-controller');});
setInterval(()=>{if(!offline&&performance.now()-latestAt>2000)$('connection').textContent='Telemetry stale · reconnecting';refreshCameras();},1000);
function mapRect(){const c=$('map'),g=state.meta?.grid;if(!g)return null;const w=c.clientWidth,h=c.clientHeight,s=Math.min(w/g.cols,h/g.rows);return {x:(w-g.cols*s)/2,y:(h-g.rows*s)/2,w:g.cols*s,h:g.rows*s,s,g};}
function point(x,y){const r=mapRect();return [r.x+(x-r.g.origin_m[0])/r.g.cell_m*r.s,r.y+r.h-(y-r.g.origin_m[1])/r.g.cell_m*r.s];}
function drawMap(){const c=$('map'),ctx=c.getContext('2d'),r=mapRect();const dpr=Math.min(devicePixelRatio,2);c.width=c.clientWidth*dpr;c.height=c.clientHeight*dpr;ctx.scale(dpr,dpr);ctx.fillStyle='#0c141b';ctx.fillRect(0,0,c.clientWidth,c.clientHeight);if(!r)return;
 if(mapImage){ctx.imageSmoothingEnabled=false;ctx.save();if(!setup){ctx.translate(r.x,r.y+r.h);ctx.scale(1,-1);ctx.drawImage(mapImage,0,0,r.w,r.h);}else ctx.drawImage(mapImage,r.x,r.y,r.w,r.h);ctx.restore();}
 const path=(pts,color,close=false)=>{if(!pts?.length)return;ctx.beginPath();pts.forEach((p,i)=>{const [x,y]=point(...p);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});if(close)ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();if(close){ctx.fillStyle=color+'22';ctx.fill();}};
 if(!setup)path(frame.route?.egress,'#f7bb6a');
 const marker=(xy,color,label,size=7)=>{const [x,y]=point(...xy);ctx.beginPath();ctx.arc(x,y,size,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#071018';ctx.lineWidth=2;ctx.stroke();ctx.font='bold 10px system-ui';ctx.fillStyle='#071018';ctx.textAlign='center';ctx.fillText(label,x,y+3);};
 if(setup){entrances.forEach((e,i)=>marker(e.start_xy_m,selected.includes(i)?colors[selected.indexOf(i)%3]:'#63717a',String(i+1),9));if(target)marker(target,'#f7bb6a','T',8);}
 else{for(const a of frame.agents||[]){if(a.x==null)continue;path(a.samples?.map(s=>[s[1],s[2]]),colors[a.id%3]);marker([a.x,a.y],colors[a.id%3],String(a.id+1));}if(state.meta?.target_xy)marker(state.meta.target_xy,'#f7bb6a','T',6);}
 if(region.length)path(region,action==='keep_out'?'#ff9292':'#76e7ca',region.length>=3);
 const approval=frame.approvals?.[0];if(approval?.intended_action?.goal_xy_m)marker(approval.intended_action.goal_xy_m,'#ff9292','!',10);
}
async function loadMap(){const key=setup&&sceneKey?`preview:${sceneKey}`:frame.map_url;if(!key||key===mapImageKey)return;mapImageKey=key;const im=new Image();im.onload=()=>{if(key===mapImageKey){mapImage=im;drawMap();}};im.onerror=()=>{if(key===mapImageKey){mapImage=null;mapImageKey='';}};im.src=setup?`/api/demo/preview.png?scene=${encodeURIComponent(sceneKey)}`:key;}
new ResizeObserver(drawMap).observe($('map'));
$('map').onclick=e=>{if(!controlled)return;const r=mapRect();if(!r)return;const b=$('map').getBoundingClientRect(),x=e.clientX-b.left,y=e.clientY-b.top;if(x<r.x||y<r.y||x>r.x+r.w||y>r.y+r.h)return;
 const xy=[r.g.origin_m[0]+(x-r.x)/r.s*r.g.cell_m,r.g.origin_m[1]+(r.y+r.h-y)/r.s*r.g.cell_m];
 if(setup){const i=entrances.findIndex(a=>{const p=point(...a.start_xy_m);return Math.hypot(p[0]-x,p[1]-y)<15;});if(i>=0){if(selected.includes(i))selected=selected.filter(a=>a!==i);else if(selected.length<+$('agent-count').value)selected.push(i);else notice('Deselect an entrance before choosing another.');}else target=xy;validate();}
 else if(action){if(region.length<32)region.push(xy);drawMap();}else{const a=(frame.agents||[]).find(a=>a.x!=null&&Math.hypot(...point(a.x,a.y).map((v,i)=>v-[x,y][i]))<15);if(a)$('agent').value=a.id;}};
document.querySelectorAll('[data-action]').forEach(b=>b.onclick=async()=>{try{const kind=b.dataset.action;if(kind==='release'){await api('directive',{kind,agent:+$('agent').value});return;}action=kind;region=[];$('apply-region').hidden=$('cancel-region').hidden=false;notice('Draw a region on the map, then Apply to confirm.');updateControls();}catch(e){notice(e.message);}});
function cancelRegion(){action=null;region=[];$('apply-region').hidden=$('cancel-region').hidden=true;updateControls();drawMap();}
on('cancel-region',cancelRegion);on('apply-region',async()=>{if(region.length<3)throw Error('Choose at least three vertices.');await api('directive',{kind:action,agent:+$('agent').value,polygon:region});notice('Directive queued; awaiting runtime validation.');cancelRegion();});
async function refreshCameras(){const all=frame.cameras||[],primary=+$('camera-agent').value;const chosen=[all.find(c=>c.agent===primary),all.find(c=>c.agent!==primary)];
 for(let i=0;i<2;i++){let d=chosen[i];const name=i?'other':'main',canvas=$(`cam-${name}`),label=$(`cam-${name}-label`);if(!d){label.textContent='Waiting for sensor frames';canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);continue;}
 let cached=cameraCache.get(d.agent);if(!replaying&&cached?.data?.t_ns>d.t_ns)d=cached.data;if(i&&cached?.fetchedAt&&performance.now()-cached.fetchedAt<500)continue;if(cached?.url!==d.url&&!cached?.busy){cameraCache.set(d.agent,{...cached,busy:true});const requestedSession=sessionKey,im=new Image();im.onload=()=>{if(requestedSession!==sessionKey)return;cameraCache.set(d.agent,{url:d.url,image:im,data:d,fetchedAt:performance.now()});paintCamera(canvas,im,d);};im.onerror=()=>cameraCache.delete(d.agent);im.src=d.url;}
 cached=cameraCache.get(d.agent);if(cached?.image)paintCamera(canvas,cached.image,cached.data);
 const age=Math.max(0,(frame.t_s||0)-d.t_ns/1e9)+(replaying?0:Math.max(0,Date.now()/1000-(frame.wall_unix_s||Date.now()/1000)));label.textContent=`Agent ${d.agent+1} · ${age.toFixed(1)}s sensor age${age>2?' · STALE':''}${d.dropped?` · ${d.dropped} display drops`:''}`;
 }}
function paintCamera(c,im,d){const scale=Math.min(devicePixelRatio,1.5);c.width=c.clientWidth*scale;c.height=c.clientHeight*scale;const ctx=c.getContext('2d'),s=Math.min(c.width/im.width,c.height/im.height),x=(c.width-im.width*s)/2,y=(c.height-im.height*s)/2;ctx.drawImage(im,x,y,im.width*s,im.height*s);if($('features').checked){ctx.fillStyle='#76e7ca';for(const p of d.pts||[]){ctx.fillRect(x+p[0]*s-1,y+p[1]*s-1,2,2);}}}
$('features').onchange=refreshCameras;$('camera-agent').onchange=refreshCameras;new ResizeObserver(refreshCameras).observe($('cameras'));
function text(tag,value,cls){const n=document.createElement(tag);n.textContent=value;if(cls)n.className=cls;return n;}
function button(label,fn){const n=text('button',label);n.onclick=()=>Promise.resolve().then(fn).catch(e=>notice(e.message));return n;}
function renderDecision(){const box=$('decision-content');const approvals=frame.approvals||[];const decisions=frame.decisions||[];const events=decisions.length?decisions:frame.events||[];
 const newKey=JSON.stringify([approvals,events.at(-1-eventIndex),eventIndex,controlled,replaying]);
 if(newKey===decisionKey){const a=approvals[eventIndex % Math.max(1,approvals.length)],label=box.querySelector('.approval .decision-kind');if(a&&label)label.textContent=`HUMAN DECISION · ${approvals.length} PENDING · ${Math.max(0,Math.ceil((a.deadline_unix_s||a.requested_unix_s+60)-Date.now()/1000))}s`;return;}
 decisionKey=newKey;box.replaceChildren();
 eventIndex=Math.min(eventIndex,Math.max(0,(approvals.length&&!replaying?approvals.length:events.length)-1));$('decision-page').textContent=`${events.length?eventIndex+1:0} / ${events.length}`;
 if(approvals.length&&!replaying){$('decision-page').textContent=`${eventIndex % approvals.length+1} / ${approvals.length}`;const a=approvals[eventIndex % approvals.length],card=text('div','','approval');card.append(text('div',`HUMAN DECISION · ${approvals.length} PENDING`,'decision-kind'),text('div',`Agent ${a.agent+1} · risk ${Number(a.risk_score||0).toFixed(2)}`,'decision-title'),text('p',`${a.orchestrator_reason||a.risk_reason||'Review this action.'} · ${Math.max(0,Math.ceil((a.deadline_unix_s||a.requested_unix_s+60)-Date.now()/1000))}s remaining`,'decision-body'));
 const acts=text('div','','decision-actions');for(const [label,verdict] of [['Approve','approved'],['Reject','rejected']]){const b=button(label,async()=>{await api('approval',{id:a.approval_id,verdict});notice('Verdict submitted; runtime remains authoritative.');});b.disabled=!controlled;acts.append(b);}acts.append(button('Evidence',()=>showDetails(a)));card.append(acts);box.append(card);return;}
 const e=events[events.length-1-eventIndex];if(!e){box.append(text('p','Waiting for observations and planner decisions.','empty'));return;}
 box.append(text('div',`${time(e.t_s)} · ${e.choice?.source||e.kind||'PLANNER'}`,'decision-kind'),text('div',e.agent!=null?`Agent ${e.agent+1} · ${e.result||'decision'}`:e.kind||'Mission event','decision-title'),text('p',[e.choice?.reason||e.text||`Command: ${e.command?.kind||'hold'} · ${e.executed?'executed':'not executed'}`,e.orchestrator?.model?`${e.orchestrator.model} · ${e.orchestrator.latency_s}s · ${e.orchestrator.tool_calls?.length||0} tool calls`:e.choice?.validation].filter(Boolean).join(' · '),'decision-body'));
 const acts=text('div','','decision-actions');acts.append(button('Inspect evidence',()=>showDetails(e)));box.append(acts);
}
on('previous',()=>{eventIndex++;renderDecision();});on('next',()=>{eventIndex=Math.max(0,eventIndex-1);renderDecision();});
function showDetails(e){const lines=JSON.stringify(e,null,2).split('\n').flatMap(l=>l.match(/.{1,75}/g)||['']);detailPages=[];for(let i=0;i<lines.length;i+=18)detailPages.push(lines.slice(i,i+18).join('\n'));detailIndex=0;renderDetails();$('details').showModal();}
function renderDetails(){$('detail-text').textContent=detailPages[detailIndex];$('detail-page').textContent=`${detailIndex+1} / ${detailPages.length}`;}
on('detail-prev',()=>{detailIndex=Math.max(0,detailIndex-1);renderDetails();});on('detail-next',()=>{detailIndex=Math.min(detailPages.length-1,detailIndex+1);renderDetails();});on('close-details',()=>$('details').close());
function stopPlayback(){replaying=false;clearTimeout(playbackTimer);$('playback').textContent='Replay recording';}
on('playback',async()=>{if(replaying){stopPlayback();return;}const r=await fetch('/api/demo/playback');if(!r.ok)throw Error('No recorded playback available.');playingFrames=(await r.text()).trim().split('\n').filter(Boolean).map(JSON.parse);replaying=true;setup=false;mapImage=null;mapImageKey='';view?.clear(sessionKey);switchView('observed');$('playback').textContent='Exit playback';notice('Recorded observations and decisions · no model calls');let i=0;const advance=()=>{if(!replaying)return;frame=playingFrames[i];renderFrame();loadMap();updateControls();if(i+1<playingFrames.length){const delay=Math.max(20,Math.min(2000,(playingFrames[i+1].t_s-frame.t_s)*1000));i++;playbackTimer=setTimeout(advance,delay);}else notice('Recorded playback complete.');};advance();});

let keyboardPoint=null;
$('map').addEventListener('keydown',e=>{
 if(!controlled||!state.meta)return;
 if(setup&&['1','2','3'].includes(e.key)){
   const i=+e.key-1;if(i>=entrances.length)return;
   selected=selected.includes(i)?selected.filter(a=>a!==i):selected.length<+$('agent-count').value?[...selected,i]:selected;
   validate();e.preventDefault();return;
 }
 if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Enter','Escape'].includes(e.key))return;
 e.preventDefault();if(e.key==='Escape'){cancelRegion();return;}
 keyboardPoint=keyboardPoint||target||state.meta.target_xy;
 const g=state.meta.grid,step=g.cell_m*(e.shiftKey?10:1);
 if(e.key==='Enter'){
   if(setup){target=[...keyboardPoint];validate();}
   else if(action){region.push([...keyboardPoint]);drawMap();}
   return;
 }
 const dx=e.key==='ArrowLeft'?-step:e.key==='ArrowRight'?step:0,dy=e.key==='ArrowDown'?-step:e.key==='ArrowUp'?step:0;
 keyboardPoint=[Math.max(g.origin_m[0],Math.min(g.origin_m[0]+(g.cols-1)*g.cell_m,keyboardPoint[0]+dx)),Math.max(g.origin_m[1],Math.min(g.origin_m[1]+(g.rows-1)*g.cell_m,keyboardPoint[1]+dy))];
 notice(`Map cursor ${keyboardPoint.map(x=>x.toFixed(2)).join(', ')} m · Enter to select`);
});

// Media is independent of the 5 Hz state stream. Only the selected camera polls
// at 10 Hz; hidden/secondary images never create an unbounded download queue.
let cameraPollBusy=false;
setInterval(async()=>{
 if(cameraPollBusy||document.hidden||replaying||state.state!=='running'||!sessionKey)return;
 cameraPollBusy=true;const requestSession=sessionKey;
 try{
   const r=await fetch(`/api/demo/cameras?agent=${$('camera-agent').value}&session=${encodeURIComponent(requestSession)}`);
   if(!r.ok)return;const d=await r.json();if(!d||requestSession!==sessionKey)return;
   frame.cameras=[...(frame.cameras||[]).filter(c=>c.agent!==d.agent),d];refreshCameras();
 }catch{}finally{cameraPollBusy=false;}
},100);
