"""Opt-in real browser test. Uses a synthetic runtime fixture, never launches Docker.
RUN_BROWSER_TESTS=1 PYTHONPATH=python pytest python/tests/console/test_demo_browser.py
"""
import json
import os
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('RUN_BROWSER_TESTS') != '1', reason='opt-in Chromium integration test')


def test_one_screen_mission_flow(tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright
    import uvicorn
    from operator_console import demo, server
    from operator_console.live import LiveRun
    import cv2

    m = demo.Demo(demo.ROOT)
    m.enabled = True
    m.base = tmp_path
    monkeypatch.setattr(demo, 'manager', m)
    monkeypatch.setattr(server, 'demo_manager', m)

    def launch(spec, key):
        chosen = m.validate(spec)
        m.session = 'browser-fixture'
        m.run = tmp_path / 'fixture-run'
        m.run.mkdir()
        m.lv = LiveRun(m.run, m.scene)
        m.state = 'running'
        m.commands = {'start': {'id':key, 'state':'effective'}}
        (m.run/'snapshots').mkdir()
        (m.run/'splat/chunks').mkdir(parents=True)
        points=b''.join(struct.pack('<ffffffBBBBBBBB',chosen['target'][0]+i*.08,chosen['target'][1]+j*.08,1,.03,.03,.03,120,230,200,255,255,128,128,128) for i in range(5) for j in range(5))
        (m.run/'splat/chunks/0_1.splat').write_bytes(points)
        (m.run/'splat/table.json').write_text(json.dumps({'version':1,'backend':'cpu_init','submaps':[{'sid':0,'chunk':'0_1.splat','gaussians':25,'bytes':len(points),'T_W_anchor':list(np.eye(4).ravel())}]}))
        image = np.full((180,320,3),(40,60,90),np.uint8)
        cv2.imwrite(str(m.run/'snapshots/cam_0_1000000000.jpg'), image)
        (m.run/'snapshots/cam_0.json').write_text(json.dumps({'agent':0,'t_ns':1000000000,'image':'cam_0_1000000000.jpg','w':320,'h':180,'pts':[[80,70],[160,100]]}))
        (m.run/'telemetry.jsonl').write_text(json.dumps({'type':'pose','agent':0,'t_s':1.,'xy':chosen['target'],'yaw':0})+'\n')
        (m.run/'approvals').mkdir()
        (m.run/'approvals/review.request.json').write_text(json.dumps({'approval_id':'review','agent':0,'risk_score':.7,'orchestrator_reason':'Fixture: review narrow passage','deadline_unix_s':time.time()+60,'intended_action':{'goal_xy_m':chosen['target']}}))
        m.tick()
        return m.status()
    monkeypatch.setattr(m,'launch',launch)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    app=uvicorn.Server(uvicorn.Config(server.app,host='127.0.0.1',port=port,log_level='error'))
    thread=threading.Thread(target=app.run,daemon=True);thread.start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=['--enable-unsafe-swiftshader'])
            page=browser.new_page(viewport={'width':1366,'height':768})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{port}',wait_until='networkidle')
            page.get_by_role('button',name='Take control').click()
            page.wait_for_function("!document.querySelector('#start').disabled",timeout=90000)
            for w,h in [(1280,600),(1280,720),(1366,768),(1440,900),(1920,1080)]:
                page.set_viewport_size({'width':w,'height':h})
                assert page.evaluate("document.body.scrollHeight <= innerHeight && document.documentElement.scrollWidth <= innerWidth")
                assert page.evaluate("[...document.querySelectorAll('main > section')].every(e=>e.scrollHeight<=e.clientHeight)")
            page.set_viewport_size({'width':1366,'height':768})
            page.get_by_role('button',name='Deploy team').click()
            page.wait_for_function("document.querySelector('#status').textContent==='running'")
            page.get_by_role('button',name='Approve',exact=True).click()
            page.wait_for_function("!document.querySelector('.approval')")
            assert json.loads((m.run/'approvals/review.json').read_text())['verdict']=='approved'
            with page.expect_response(lambda r: r.url.endswith('/api/demo/hold') and r.request.method == 'POST') as held:
                page.get_by_role('button',name='Hold team').click()
            assert held.value.status == 200
            assert list((m.run/'directives_inbox').glob('*.json'))
            page.wait_for_function("document.querySelector('#geometry-label').textContent.includes('1 submaps')")
            assert 'Observed reconstruction' in page.locator('#view-title').inner_text()
            assert 'Observed map' in page.locator('#map-title').inner_text()
            for w,h in [(1280,600),(1280,720),(1440,900)]:
                page.set_viewport_size({'width':w,'height':h})
                assert page.evaluate("[...document.querySelectorAll('main > section')].every(e=>e.scrollHeight<=e.clientHeight)")
            page.get_by_role('button',name='End run',exact=True).click()
            page.wait_for_function("document.querySelector('#status').textContent==='stopping'")
            assert (m.run/'stop.request.json').exists()
            with m.lock:
                m.state='completed'
            page.get_by_role('button',name='Replay recording').click()
            page.wait_for_function("document.querySelector('#map-mode').textContent==='RECORDED'")
            page.reload(wait_until='domcontentloaded')
            page.wait_for_function("document.querySelector('#map-title').textContent==='Observed map'")
            page.wait_for_function("document.querySelector('#geometry-label').textContent.includes('1 submaps')")
            assert page.locator('#setup').is_hidden()
            assert not errors
            browser.close()
    finally:
        app.should_exit=True
        thread.join(timeout=5)
