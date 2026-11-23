"""The dashboard: live view, what's in view now, activity log, vehicles, recordings, phone alerts."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


class LiveState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg: bytes = b""
        self.people: list[dict] = []
        self.vehicles: list[dict] = []
        self.stream_fps = 0.0
        self.lag = 0.0
        self.recording = ""
        self.zones: list[dict] = []
        self.raw_jpeg: bytes = b""   # picture without boxes, for drawing zones


class App:
    """Everything the web page needs to read."""

    def __init__(self, state: LiveState, db, notifier, recordings_dir: str, camera: str, keep_days: int,
                 zone_store=None):
        self.state, self.db, self.notifier, self.zone_store = state, db, notifier, zone_store
        self.recordings_dir, self.camera, self.keep_days = Path(recordings_dir), camera, keep_days


def serve(app: App, port: int) -> None:
    handler = type("Handler", (Handler,), {"app": app})
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    app: App

    def log_message(self, *a):
        pass

    # ------------------------------------------------------------ routing
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        limit = max(1, min(500, int(q.get("limit", "100") or 100)))
        a = self.app
        if url.path == "/":
            return self._send(200, "text/html; charset=utf-8", DASHBOARD.encode())
        if url.path == "/stream.mjpg":
            return self._stream()
        if url.path == "/frame.jpg":
            with a.state.lock:
                jpeg = a.state.raw_jpeg or a.state.jpeg
            return self._send(200 if jpeg else 503, "image/jpeg", jpeg)
        if url.path == "/api/zones":
            return self._json(a.zone_store.as_dicts() if a.zone_store else [])
        if url.path == "/snapshot.jpg":
            with a.state.lock:
                jpeg = a.state.jpeg
            return self._send(200 if jpeg else 503, "image/jpeg", jpeg)
        if url.path == "/api/status":
            today = datetime.now().strftime("%Y-%m-%d")
            counts = a.db.query(
                "SELECT SUM(kind='person') AS people, SUM(kind='vehicle') AS vehicles, "
                "SUM(kind='vehicle' AND status='unknown') AS unknown FROM events WHERE ts >= ?", today)[0]
            with a.state.lock:
                body = {"camera": a.camera, "people": a.state.people, "vehicles": a.state.vehicles,
                        "stream_fps": round(a.state.stream_fps, 1), "lag": round(a.state.lag, 1),
                        "recording": a.state.recording, "today": {k: v or 0 for k, v in counts.items()}}
            return self._json(body)
        if url.path == "/api/events":
            kind = q.get("kind", "")
            sql = "SELECT * FROM events" + (" WHERE kind = ?" if kind else "") + " ORDER BY id DESC LIMIT ?"
            return self._json(a.db.query(sql, *([kind] if kind else []), limit))
        if url.path == "/api/vehicles":
            return self._json(a.db.query(
                "SELECT * FROM events WHERE kind='vehicle' AND plate IS NOT NULL AND plate != '' "
                "ORDER BY id DESC LIMIT ?", limit))
        if url.path == "/api/recordings":
            rows = a.db.query("SELECT * FROM recordings ORDER BY id DESC LIMIT ?", limit)
            for r in rows:
                r["exists"] = (a.recordings_dir / (r["file"] or "")).is_file()
            return self._json([r for r in rows if r["exists"]])
        if url.path == "/api/clip-events":
            return self._json(a.db.query("SELECT * FROM events WHERE clip = ? ORDER BY id", q.get("file", "")))
        if url.path == "/api/alerts":
            n = a.notifier
            return self._json({"server": n.phone_server, "topic": n.topic, "subscribe_url": n.subscribe_url,
                               "qr_svg": n.qr_svg(), "keep_days": a.keep_days,
                               "recent": a.db.query("SELECT * FROM alerts ORDER BY id DESC LIMIT 20")})
        if url.path.startswith("/rec/"):
            return self._send_file(url.path[len("/rec/"):])
        return self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if urlparse(self.path).path == "/api/zones" and self.app.zone_store is not None:
            try:
                n = int(self.headers.get("Content-Length", "0"))
                zones = json.loads(self.rfile.read(min(n, 200_000)) or b"[]")
                return self._json(self.app.zone_store.save(zones))
            except Exception as e:
                return self._send(400, "text/plain", f"bad zones: {e}".encode())
        if urlparse(self.path).path == "/api/test-alert":
            ok = self.app.notifier.send("Test alert", f"Farm cameras are connected to this phone – "
                                        f"{datetime.now():%d %b %H:%M:%S}", tags="white_check_mark",
                                        priority="default")
            return self._json({"ok": ok})
        return self._send(404, "text/plain", b"not found")

    # ------------------------------------------------------------ helpers
    def _json(self, obj):
        self._send(200, "application/json", json.dumps(obj).encode())

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last = b""
        try:
            while True:
                with self.app.state.lock:
                    jpeg = self.app.state.jpeg
                if jpeg and jpeg is not last:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                    last = jpeg
                time.sleep(0.04)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_file(self, rel: str):
        """Serves a recording, with Range support so browsers can play and seek video."""
        base = self.app.recordings_dir.resolve()
        f = (base / unquote(rel)).resolve()
        if base not in f.parents or not f.is_file() or f.suffix not in (".mp4", ".jpg"):
            return self._send(404, "text/plain", b"not found")
        size = f.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes=") and size:
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b or 0))
            end = min(int(b), size - 1) if (a and b) else end
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "video/mp4" if f.suffix == ".mp4" else "image/jpeg")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        self.end_headers()
        try:
            with open(f, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


DASHBOARD = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Farm Security</title>
<style>
:root{--bg:#0f1115;--panel:#181b22;--line:#262b35;--text:#e8eaed;--muted:#8b93a1;--accent:#5aa9ff;
--ok:#3fbf6f;--bad:#ff5252;--warn:#f5c542;--person:#4fc3f7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,sans-serif}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:18px;margin:0 12px 0 0}.pill{padding:3px 10px;border-radius:99px;background:#222733;color:var(--muted);font-size:12px}
.pill b{color:var(--text)}.rec{background:#3a1212;color:#ff6b6b;font-weight:600;animation:blink 1.2s infinite}
@keyframes blink{50%{opacity:.55}}.spacer{flex:1}
main{padding:14px 16px;display:grid;gap:14px;grid-template-columns:minmax(0,3fr) minmax(280px,1fr);max-width:1800px;margin:auto}
@media(max-width:1000px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.panel h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0;padding:10px 14px;border-bottom:1px solid var(--line)}
#live img{width:100%;display:block;background:#000;aspect-ratio:16/9}
#now{padding:10px 14px;display:flex;flex-direction:column;gap:8px;max-height:70vh;overflow:auto}
.card{border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:#1d212a}
.card .t{display:flex;justify-content:space-between;align-items:center;font-weight:600}
.bar{height:5px;background:#2a303b;border-radius:3px;margin-top:6px}.bar i{display:block;height:100%;border-radius:3px;background:var(--person)}
.muted{color:var(--muted)}.small{font-size:12px}
.plate{display:inline-block;font:700 15px/1 "Arial Narrow",Arial,sans-serif;letter-spacing:.06em;background:#f7d117;color:#111;
padding:4px 7px;border-radius:3px;border:1px solid #000}
.st-known{color:var(--ok)}.st-unknown{color:var(--bad);font-weight:700}.st-checking{color:var(--warn)}.st-no-plate{color:var(--muted)}
.full{grid-column:1/-1}
.tabs{display:flex;gap:2px;padding:0 8px;border-bottom:1px solid var(--line);overflow-x:auto}
.tabs button{background:none;border:0;color:var(--muted);padding:11px 14px;font:inherit;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap}
.tabs button.on{color:var(--text);border-color:var(--accent)}
.tab{display:none;padding:12px 14px}.tab.on{display:block}
.tablewrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:500;position:sticky;top:0;background:var(--panel)}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.clip{background:#1d212a;border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer}
.clip img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;background:#000}.clip div{padding:7px 9px}
.btn{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:9px 14px;font:inherit;cursor:pointer}
.btn:disabled{opacity:.5}.qr{background:#fff;border-radius:8px;padding:6px;width:220px}
.qr svg{width:100%;height:auto;display:block}
.steps{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start}.steps ol{margin:0;padding-left:18px;max-width:520px}.steps li{margin:6px 0}
code{background:#222733;padding:2px 6px;border-radius:4px;user-select:all}
dialog{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:0;max-width:min(1200px,96vw);width:96vw}
dialog video{width:100%;display:block;background:#000;max-height:70vh}dialog .body{padding:12px 14px}
dialog::backdrop{background:rgba(0,0,0,.7)}.x{float:right;background:none;border:0;color:var(--text);font-size:22px;cursor:pointer}
.empty{color:var(--muted);padding:14px 0}
.doing{margin-top:4px;font-size:13px}.alarm{color:#fff;background:var(--bad);border-radius:4px;padding:1px 6px;font-size:12px;font-weight:600;margin-left:4px}
.beh{color:var(--warn)}.beh.alert{color:var(--bad);font-weight:600}
.zedit{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}.zpic{position:relative;flex:1 1 560px;max-width:960px;cursor:crosshair}
.zpic img{width:100%;display:block;border-radius:6px;background:#000}.zpic svg{position:absolute;inset:0;width:100%;height:100%}
.zform{flex:0 1 300px;display:flex;flex-direction:column;gap:8px}.zform input,.zform select{background:#222733;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:8px;font:inherit}
.btn2{background:#2a303b;color:var(--text);border:0;border-radius:6px;padding:8px 12px;font:inherit;cursor:pointer}
.zlist div{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--line)}
</style></head><body>
<header><h1>🐄🐖🐔 Farm Security</h1>
<span class="pill" id="cam">camera …</span><span class="pill" id="delay">delay …</span>
<span class="pill rec" id="rec" hidden>● REC</span><span class="spacer"></span>
<span class="pill">Today: <b id="tp">0</b> people · <b id="tv">0</b> vehicles · <b id="tu" style="color:var(--bad)">0</b> unknown</span>
</header>
<main>
 <section class="panel" id="live"><h2>Live view</h2><img src="/stream.mjpg" alt="Live camera"></section>
 <section class="panel"><h2>In view now</h2><div id="now"><div class="empty">Nobody in view</div></div></section>
 <section class="panel full">
  <div class="tabs">
   <button data-tab="activity" class="on">Activity log</button><button data-tab="vehicles">Vehicles &amp; plates</button>
   <button data-tab="recordings">Recordings</button><button data-tab="zones">Zones</button><button data-tab="alerts">Phone alerts</button>
  </div>
  <div class="tab on" id="activity"><div class="tablewrap"><table><thead><tr><th>Time</th><th>What</th><th>Confidence</th>
   <th>Plate</th><th>Details / doing</th><th>Status</th><th>Clip</th></tr></thead><tbody></tbody></table></div></div>
  <div class="tab" id="vehicles"><div class="tablewrap"><table><thead><tr><th>Time</th><th>Plate</th><th>Status</th><th>Country</th>
   <th>Region registered</th><th>Plate issued</th><th>Type</th><th>Colour (approx)</th><th>Vehicle conf.</th><th>Plate conf.</th><th>Clip</th>
   </tr></thead><tbody></tbody></table></div>
   <p class="small muted">Region and issue date are worked out from the plate itself (UK plates since 2001: first letter = region, numbers = when issued).
   Private plates can move between cars, so the date is for the plate, not necessarily the car.</p></div>
  <div class="tab" id="recordings"><p class="small muted" id="recinfo"></p><div class="grid"></div></div>
  <div class="tab" id="zones"><div class="zedit">
   <div class="zpic" id="zpic"><img id="zimg" alt="camera picture"><svg id="zsvg" viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div>
   <div class="zform"><b>Draw a zone</b><span class="small muted">Click on the picture to add corners (3 or more), then name it and save.
    Someone is "in" a zone when their feet are inside it.</span>
    <input id="zname" placeholder="Name, e.g. Lairage, Back fence">
    <select id="ztype"><option value="restricted">Restricted – alert when someone enters</option><option value="watch">Watch – log only</option></select>
    <input id="zhours" placeholder="When: always, or e.g. 18:00-07:00" value="always">
    <div style="display:flex;gap:6px;flex-wrap:wrap"><button class="btn" id="zsave">Save zone</button><button class="btn2" id="zundo">Undo point</button>
     <button class="btn2" id="zclear">Clear</button><button class="btn2" id="zrefresh">New picture</button></div>
    <span id="zmsg" class="small"></span><b style="margin-top:10px">Zones</b><div class="zlist" id="zlist"></div></div>
  </div></div>
  <div class="tab" id="alerts"><div class="steps">
   <div class="qr" id="qr"></div>
   <ol><li>Install the free <b>ntfy</b> app (App Store / Google Play).</li>
    <li>In the app tap <b>+</b>. Topic name: <code id="topic">…</code></li>
    <li id="serverstep">Leave <i>Use another server</i> unticked.</li>
    <li>Tap <b>Subscribe</b>, then press the button below – your phone should ping.</li>
    <li class="small muted">Or scan the QR code with the phone camera.</li></ol>
   <div><button class="btn" id="testbtn">Send test alert</button><p id="testres" class="small"></p></div>
  </div><h3 style="font-size:14px;margin:18px 0 6px">Recent alerts</h3>
   <div class="tablewrap"><table><thead><tr><th>Time</th><th>Alert</th><th>Details</th><th>Delivered</th></tr></thead><tbody id="alertrows"></tbody></table></div>
  </div>
 </section>
</main>
<dialog id="player"><button class="x" onclick="player.close()">×</button><video controls playsinline></video>
<div class="body"><b id="ptitle"></b> <a id="praw" target="_blank">Original 4K (no boxes) ↗</a><div id="pevents" class="small" style="margin-top:8px"></div></div></dialog>
<script>
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=v=>v==null?'':Math.round(v*100)+'%', tm=ts=>ts?ts.replace('T',' ').slice(5,19):'';
const plate=p=>p?`<span class="plate">${esc(p)}</span>`:'';
const statusTxt={known:'✓ Allowed',unknown:'⚠ UNKNOWN',checking:'checking…','no-plate':'no plate seen'};
const st=(s,o)=>s?`<span class="st-${esc(s)}">${statusTxt[s]||esc(s)}${o?' – '+esc(o):''}</span>`:'';
const clipLink=c=>c?`<a href="#" onclick="play('${esc(c)}');return false">▶ play</a>`:'';
const details=e=>[e.country,e.region,e.registered&&('issued '+e.registered),e.colour].filter(Boolean).map(esc).join(' · ');
async function j(u,o){const r=await fetch(u,o);return r.json()}

async function status(){try{const s=await j('/api/status');
 $('#cam').innerHTML='camera <b>'+s.stream_fps+'</b> fps';$('#delay').innerHTML='delay <b>'+s.lag+'</b> s';
 $('#rec').hidden=!s.recording;$('#tp').textContent=s.today.people;$('#tv').textContent=s.today.vehicles;$('#tu').textContent=s.today.unknown;
 const cards=[...s.people.map(p=>`<div class="card"${(p.alerts||[]).length||(p.restricted||[]).length?' style="border-color:var(--bad)"':''}><div class="t"><span style="color:var(--person)">Person ${p.number}${(p.alerts||[]).map(a=>'<span class="alarm">'+esc(a)+'</span>').join('')}${(p.restricted||[]).map(z=>'<span class="alarm">in '+esc(z)+'</span>').join('')}</span><span>${pct(p.score)}</span></div>
  <div class="bar"><i style="width:${pct(p.score)}"></i></div>
  <div class="doing">${esc(p.doing||'')}${(p.zones||[]).length?' · <span class="muted">in '+(p.zones||[]).map(esc).join(', ')+'</span>':''}${p.seconds?' <span class="muted small">· '+p.seconds+'s in view</span>':''}</div></div>`),
  ...s.vehicles.map(v=>`<div class="card"><div class="t"><span>Vehicle ${v.number} <span class="muted small">${esc(v.kind)}</span></span><span>${pct(v.score)}</span></div>
  <div style="margin-top:6px">${plate(v.plate)} ${st(v.status,v.owner)}</div>
  <div class="small muted" style="margin-top:4px">${details(v)}${v.plate_conf?' · plate '+pct(v.plate_conf):''}</div></div>`)];
 $('#now').innerHTML=cards.join('')||'<div class="empty">Nobody in view</div>';}catch(e){}}

async function activity(){const rows=await j('/api/events?limit=150');
 $('#activity tbody').innerHTML=rows.map(e=>`<tr><td>${tm(e.ts)}</td>
  <td>${e.kind==='person'?'<span style="color:var(--person)">Person '+e.number+'</span>':e.kind==='behaviour'?'<span style="color:var(--person)">Person '+e.number+'</span> <span class="beh'+(e.status==='alert'?' alert':'')+'">→ '+esc(e.label)+'</span>':'Vehicle '+e.number+' <span class="muted">'+esc(e.label)+'</span>'}</td>
  <td>${pct(e.confidence)}</td><td>${plate(e.plate)}</td><td class="small">${e.kind==='person'?esc(e.actions||''):e.kind==='behaviour'?esc(e.details||''):details(e)}</td>
  <td>${e.kind==='vehicle'?st(e.status,e.owner):''}</td><td>${clipLink(e.clip)}</td></tr>`).join('')||'<tr><td colspan=7 class="empty">Nothing seen yet</td></tr>';}

async function vehicles(){const rows=await j('/api/vehicles?limit=200');
 $('#vehicles tbody').innerHTML=rows.map(e=>`<tr><td>${tm(e.ts)}</td><td>${plate(e.plate)}</td><td>${st(e.status,e.owner)}</td>
  <td>${esc(e.country)}</td><td>${esc(e.region)}</td><td>${esc(e.registered)}</td><td>${esc(e.label)}</td><td>${esc(e.colour)}</td>
  <td>${pct(e.confidence)}</td><td>${pct(e.plate_conf)}</td><td>${clipLink(e.clip)}</td></tr>`).join('')||'<tr><td colspan=11 class="empty">No plates read yet</td></tr>';}

async function recordings(){const rows=await j('/api/recordings?limit=120');
 $('#recordings .grid').innerHTML=rows.map(r=>`<div class="clip" onclick="play('${esc(r.file)}')">
  ${r.thumb?`<img loading="lazy" src="/rec/${esc(r.thumb)}">`:'<img>'}
  <div><b>${tm(r.start)}</b>${r.end?'':' <span style="color:var(--bad)">● recording</span>'}<br>
  <span class="small muted">${r.people} people · ${r.vehicles} vehicles ${r.plates?'· '+esc(r.plates):''}</span></div></div>`).join('')
  ||'<div class="empty">No recordings yet – they appear when a person or vehicle is seen.</div>';}

async function alerts(){const a=await j('/api/alerts');
 $('#qr').innerHTML=a.qr_svg;$('#topic').textContent=a.topic;
 $('#serverstep').innerHTML=a.server==='https://ntfy.sh'?'Leave <i>Use another server</i> <b>unticked</b>.':
  'Tick <i>Use another server</i> and enter <code>'+esc(a.server)+'</code> (phone must be on the farm Wi-Fi).';
 $('#recinfo').textContent='Recordings show the boxes and labels. Each also has the original 4K video. Kept '+a.keep_days+' days.';
 $('#alertrows').innerHTML=a.recent.map(r=>`<tr><td>${tm(r.ts)}</td><td>${esc(r.title)}</td><td class="small">${esc(r.body)}</td>
  <td>${r.ok?'<span class="st-known">✓</span>':'<span class="st-unknown" title="'+esc(r.error)+'">✗ failed</span>'}</td></tr>`).join('')
  ||'<tr><td colspan=4 class="empty">No alerts yet</td></tr>';}

async function play(file){const p=$('#player');p.querySelector('video').src='/rec/'+file;
 $('#ptitle').textContent=file.replace('.mp4','').replace('/',' ');$('#praw').href='/rec/'+file.replace('.mp4','_4k.mp4');
 const ev=await j('/api/clip-events?file='+encodeURIComponent(file));
 $('#pevents').innerHTML=ev.map(e=>`${tm(e.ts)} – ${e.kind==='person'?'Person '+e.number+(e.actions?' ('+esc(e.actions)+')':''):e.kind==='behaviour'?'Person '+e.number+' → '+esc(e.label):'Vehicle '+e.number+' '+plate(e.plate)+' '+st(e.status,e.owner)} (${pct(e.confidence)})`).join('<br>');
 p.showModal();p.querySelector('video').play().catch(()=>{});}
$('#player').addEventListener('close',()=>{const v=$('#player video');v.pause();v.removeAttribute('src');v.load()});

$('#testbtn').onclick=async()=>{const b=$('#testbtn');b.disabled=true;$('#testres').textContent='Sending…';
 try{const r=await j('/api/test-alert',{method:'POST'});$('#testres').innerHTML=r.ok?'<span class="st-known">Sent ✓ – check your phone</span>':
 '<span class="st-unknown">Failed – see Recent alerts below</span>';}catch(e){$('#testres').textContent='Failed'}b.disabled=false;alerts();};

let zoneList=[],draft=[];
function zdraw(){const svg=$('#zsvg');const poly=(pts,c,fill)=>`<polygon points="${pts.map(p=>p.join(',')).join(' ')}" fill="${fill}" stroke="${c}" stroke-width="0.004"/>`;
 svg.innerHTML=zoneList.map(z=>poly(z.points,z.type==='restricted'?'#ff5252':'#f5c542',z.type==='restricted'?'rgba(255,82,82,.18)':'rgba(245,197,66,.15)')).join('')
  +(draft.length?poly(draft,'#5aa9ff','rgba(90,169,255,.2)')+draft.map(p=>`<circle cx="${p[0]}" cy="${p[1]}" r="0.008" fill="#5aa9ff"/>`).join(''):'');
 $('#zlist').innerHTML=zoneList.map((z,i)=>`<div><span><b>${esc(z.name)}</b> <span class="small muted">${z.type} · ${esc(z.hours)}</span></span>
  <button class="btn2" onclick="zdel(${i})">Delete</button></div>`).join('')||'<span class="small muted">No zones yet</span>';}
async function zones(first){if(first!==false){$('#zimg').src='/frame.jpg?'+Date.now()}zoneList=await j('/api/zones');zdraw();}
async function zput(list,msg){try{zoneList=await j('/api/zones',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(list)});
 $('#zmsg').innerHTML='<span class="st-known">'+msg+'</span>';}catch(e){$('#zmsg').innerHTML='<span class="st-unknown">Could not save</span>'}zdraw();}
function zdel(i){if(confirm('Delete zone "'+zoneList[i].name+'"?')){const l=zoneList.slice();l.splice(i,1);zput(l,'Deleted')}}
$('#zpic').addEventListener('click',e=>{const r=$('#zimg').getBoundingClientRect();draft.push([+((e.clientX-r.left)/r.width).toFixed(4),+((e.clientY-r.top)/r.height).toFixed(4)]);zdraw();});
$('#zundo').onclick=()=>{draft.pop();zdraw()};$('#zclear').onclick=()=>{draft=[];zdraw()};$('#zrefresh').onclick=()=>zones();
$('#zsave').onclick=()=>{const name=$('#zname').value.trim();if(draft.length<3){$('#zmsg').textContent='Click at least 3 corners on the picture first';return}
 if(!name){$('#zmsg').textContent='Give the zone a name';return}
 zput([...zoneList,{name,type:$('#ztype').value,hours:$('#zhours').value||'always',points:draft}],'Saved "'+name+'" – active straight away');draft=[];$('#zname').value='';};

const loaders={activity,vehicles,recordings,zones:()=>zones(false),alerts};let current='activity';
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button,.tab').forEach(x=>x.classList.remove('on'));
 b.classList.add('on');current=b.dataset.tab;$('#'+current).classList.add('on');current==='zones'?zones():loaders[current]();});
status();activity();setInterval(status,1000);setInterval(()=>loaders[current](),4000);
</script></body></html>"""
