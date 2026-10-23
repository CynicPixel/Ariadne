import socket, struct, json, os, sys, collections
path = sys.argv[1]
try: os.unlink(path)
except FileNotFoundError: pass
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(path); os.chmod(path, 0o777); srv.listen(1)
conn, _ = srv.accept()
def rd(n):
    b = b''
    while len(b) < n:
        c = conn.recv(n - len(b))
        if not c: return None
        b += c
    return b
counts = collections.Counter(); first_kf = None; bad = 0
while True:
    h = rd(4)
    if h is None: break
    if h != b'ARL1': bad += 1; break
    tl = struct.unpack('<I', rd(4))[0]; topic = rd(tl).decode()
    pl = struct.unpack('<I', rd(4))[0]; payload = rd(pl)
    counts[topic] += 1
    if topic == '#kf' and first_kf is None:
        jl = struct.unpack('<I', payload[:4])[0]
        j = json.loads(payload[4:4+jl])
        expect = 4 + jl + j['zw']*j['zh']*2*2 + j['w']*j['h']*3
        first_kf = {'json': j, 'payload_ok': expect == pl, 'payload': pl, 'expect': expect}
    if topic == '#t_w_o' and counts['#t_w_o'] <= 3:
        print('t_w_o:', payload.decode()[:120], flush=True)
print('counts:', dict(counts), 'bad:', bad, flush=True)
if first_kf:
    k = first_kf['json']
    print('kf json keys:', sorted(k), flush=True)
    print('agent', k['agent'], 'K', k['K'], 'Kz', k['Kz'], 'w/h', k['w'], k['h'], 'zw/zh', k['zw'], k['zh'], flush=True)
    print('payload_ok:', first_kf['payload_ok'], first_kf['payload'], first_kf['expect'], flush=True)
