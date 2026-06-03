from flask import Flask, jsonify, request
import carla
import math
import time
import threading
import json
import os

app = Flask(__name__)

# Global state
client = None
world  = None
vehicle = None
is_running = True

# Manual control state
manual_mode = False
manual_throttle = 0.0
manual_steer    = 0.0
manual_brake    = 0.0
manual_reverse  = False
# Absolute path — same as advanced_drive.py reads
MANUAL_CONTROL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manual_control.json')

def write_manual_state():
    """Write manual control state to file for advanced_drive.py to read"""
    try:
        with open(MANUAL_CONTROL_FILE, 'w') as f:
            json.dump({
                'manual_mode': manual_mode,
                'throttle':    manual_throttle,
                'steer':       manual_steer,
                'brake':       manual_brake,
                'reverse':     manual_reverse
            }, f)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
def connect_to_carla():
    global client, world, vehicle
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print("[OK] Connected to CARLA server")

        while vehicle is None and is_running:
            actors = world.get_actors().filter('vehicle.*')
            for actor in actors:
                if actor.attributes.get('role_name') == 'hero':
                    vehicle = actor
                    print(f"[OK] Found hero vehicle: {vehicle.type_id}")
                    break
            if vehicle is None:
                print("[..] Waiting for hero vehicle...")
                time.sleep(1)
    except Exception as e:
        print(f"[ERR] Failed to connect: {e}")


def get_traffic_light_state(veh):
    """
    Returns the traffic light state for OUR lane only.
    1. If vehicle is AT a light — use CARLA's built-in (100% accurate).
    2. Otherwise scan nearby lights and pick the one AHEAD of the car
       using the vehicle's forward vector (dot product filter).
       This avoids showing RED from cross-traffic signals.
    """
    try:
        # Priority 1 — most accurate: CARLA says we're at a light
        if veh.is_at_traffic_light():
            tl = veh.get_traffic_light()
            if tl:
                state = tl.get_state()
                if state == carla.TrafficLightState.Red:    return "RED"
                if state == carla.TrafficLightState.Yellow: return "YELLOW"
                if state == carla.TrafficLightState.Green:  return "GREEN"

        # Priority 2 — find the light that is AHEAD of us (not beside/behind)
        veh_transform = veh.get_transform()
        veh_loc       = veh.get_location()

        # Forward unit vector of the vehicle
        fwd = veh_transform.get_forward_vector()

        best_tl   = None
        best_dist = 999.0

        tl_actors = veh.get_world().get_actors().filter('traffic.traffic_light*')
        for tl in tl_actors:
            tl_loc = tl.get_location()
            dist   = tl_loc.distance(veh_loc)
            if dist > 60.0:          # only care about lights within 60 m
                continue

            # Direction vector from vehicle to traffic light (normalized)
            dx = tl_loc.x - veh_loc.x
            dy = tl_loc.y - veh_loc.y
            length = math.sqrt(dx*dx + dy*dy) or 1.0
            dx /= length;  dy /= length

            # Dot product with forward vector — positive = in front
            dot = dx * fwd.x + dy * fwd.y
            if dot < 0.3:            # light is not ahead of us — skip
                continue

            if dist < best_dist:
                best_dist = dist
                best_tl   = tl

        if best_tl:
            state = best_tl.get_state()
            if state == carla.TrafficLightState.Red:    return "RED"
            if state == carla.TrafficLightState.Yellow: return "YELLOW"
            if state == carla.TrafficLightState.Green:  return "GREEN"

    except Exception:
        pass
    return "NONE"


# ──────────────────────────────────────────────────────────────────────
@app.route('/control', methods=['POST'])
def set_control():
    """Receive manual control commands from dashboard"""
    global manual_mode, manual_throttle, manual_steer, manual_brake, manual_reverse
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No JSON'}), 400

    if 'manual_mode' in data:
        manual_mode = bool(data['manual_mode'])

    if manual_mode:
        manual_throttle = float(data.get('throttle', 0.0))
        manual_steer    = float(data.get('steer',    0.0))
        manual_brake    = float(data.get('brake',    0.0))
        manual_reverse  = bool(data.get('reverse',   False))

    write_manual_state()
    return jsonify({'status': 'ok', 'manual_mode': manual_mode})


@app.route('/mode')
def get_mode():
    return jsonify({'manual_mode': manual_mode})

@app.route('/weather', methods=['POST'])
def trigger_weather():
    """Write a flag file to tell advanced_drive to cycle the weather."""
    try:
        with open('weather_cmd.txt', 'w') as f:
            f.write("1")
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)})


# ──────────────────────────────────────────────────────────────────────
@app.route('/data')
def get_data():
    if vehicle is None:
        return jsonify({"error": "Vehicle not found"})
    try:
        control      = vehicle.get_control()
        velocity     = vehicle.get_velocity()
        transform    = vehicle.get_transform()
        acceleration = vehicle.get_acceleration()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        traffic = get_traffic_light_state(vehicle)

        telemetry = {}
        if os.path.exists('telemetry.json'):
            try:
                with open('telemetry.json', 'r') as f:
                    telemetry = json.load(f)
            except Exception:
                pass

        return jsonify({
            "speed":        round(speed, 1),
            "steer":        round(control.steer, 2),
            "throttle":     round(control.throttle, 2),
            "brake":        round(control.brake, 2),
            "hand_brake":   control.hand_brake,
            "reverse":      control.reverse,
            "gear":         control.gear,
            "traffic_light": traffic,
            "location": {
                "x": round(transform.location.x, 1),
                "y": round(transform.location.y, 1),
            },
            "heading":      round(transform.rotation.yaw, 1),
            "acceleration": round(math.sqrt(acceleration.x**2 + acceleration.y**2), 2),
            "ai": telemetry
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/health')
def health():
    if vehicle:
        return jsonify({"status": "ok", "vehicle": vehicle.type_id})
    return jsonify({"status": "waiting"})


# ──────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>CARLA AI Autonomous Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:       #0a0510;
    --card:     #170a29;
    --border:   #ff007f;
    --cyan:     #00f0ff;
    --green:    #00ff9d;
    --yellow:   #ffe600;
    --orange:   #ff5e00;
    --red:      #ff1744;
    --purple:   #b000ff;
    --text:     #f8f8ff;
    --dim:      #8c7b99;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── HEADER ── */
  header {
    background: linear-gradient(90deg, #120024, #2a005c, #120024);
    border-bottom: 2px solid var(--cyan);
    padding: 14px 30px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .logo { font-family:'Orbitron',sans-serif; font-size:1.4rem; color:var(--cyan); letter-spacing:3px; }
  .logo span { color:#fff; }
  .live-badge {
    display:flex; align-items:center; gap:8px;
    background:rgba(0,229,255,.1); border:1px solid var(--cyan);
    padding:6px 14px; border-radius:20px; font-size:.8rem; color:var(--cyan);
  }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--green); animation:blink 1s infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

  /* ── GRID ── */
  .grid {
    display: grid;
    grid-template-columns: 260px 1fr 260px;
    gap: 18px;
    padding: 20px 24px;
    max-width: 1400px;
    margin: 0 auto;
  }
  @media(max-width:900px){
    .grid { grid-template-columns:1fr; }
  }

  /* ── CARD ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    position: relative;
    overflow: hidden;
  }
  .card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  }
  .card-title {
    font-family:'Orbitron',sans-serif; font-size:.7rem; letter-spacing:2px;
    color:var(--dim); margin-bottom:14px; text-transform:uppercase;
  }

  /* ── SPEEDOMETER ── */
  .speed-wrap { text-align:center; padding:10px 0; }
  .speed-num {
    font-family:'Orbitron',sans-serif; font-size:5rem; font-weight:900;
    color:var(--green); line-height:1;
    text-shadow: 0 0 30px var(--green);
    transition: color .3s;
  }
  .speed-unit { font-size:.9rem; color:var(--dim); margin-top:4px; }
  .speed-target { font-size:.8rem; color:var(--cyan); margin-top:6px; }

  /* ── TRAFFIC LIGHT ── */
  .tl-wrap { display:flex; flex-direction:column; align-items:center; gap:0; }
  .tl-housing {
    background:#0a0a0a; border:3px solid #1a1a1a; border-radius:18px;
    padding:14px 10px; display:flex; flex-direction:column; gap:12px;
    box-shadow: inset 0 0 20px rgba(0,0,0,.8), 0 0 30px rgba(0,229,255,.1);
  }
  .bulb {
    width:70px; height:70px; border-radius:50%;
    background:#1a1a1a; border:2px solid #111;
    transition: background .3s, box-shadow .3s;
    position:relative;
  }
  .bulb.on-red    { background:var(--red);    box-shadow:0 0 40px var(--red),    0 0 80px rgba(255,23,68,.4); }
  .bulb.on-yellow { background:var(--yellow); box-shadow:0 0 40px var(--yellow), 0 0 80px rgba(255,234,0,.4); }
  .bulb.on-green  { background:var(--green);  box-shadow:0 0 40px var(--green),  0 0 80px rgba(0,230,118,.4); }
  .tl-label {
    margin-top:10px; font-family:'Orbitron',sans-serif;
    font-size:.85rem; letter-spacing:2px; color:var(--text); text-align:center;
  }
  #tl-text { font-size:1.3rem; font-weight:700; }

  /* ── MODE BADGE ── */
  .mode-badge {
    display:inline-block; padding:8px 20px; border-radius:30px;
    font-family:'Orbitron',sans-serif; font-size:.85rem; letter-spacing:2px;
    font-weight:700; border:2px solid; margin-bottom:16px;
    transition: all .3s;
  }

  /* ── BARS ── */
  .bar-row { margin-bottom:12px; }
  .bar-label { font-size:.75rem; color:var(--dim); margin-bottom:4px; display:flex; justify-content:space-between; }
  .bar-track {
    height:10px; background:#0a1520; border-radius:6px; overflow:hidden;
    border:1px solid var(--border);
  }
  .bar-fill { height:100%; border-radius:6px; transition:width .3s ease; }
  .bar-throttle .bar-fill { background:linear-gradient(90deg,#00897b,var(--green)); }
  .bar-brake    .bar-fill { background:linear-gradient(90deg,#bf360c,var(--red)); }
  .bar-steer    .bar-fill { background:linear-gradient(90deg,#283593,var(--cyan)); }

  /* ── LANE METER ── */
  .lane-track {
    height:16px; background:#0a1520; border-radius:8px;
    border:1px solid var(--border); position:relative; margin-bottom:8px;
  }
  .lane-center-line {
    position:absolute; left:50%; top:0; bottom:0; width:2px;
    background:rgba(255,255,255,.2);
  }
  .lane-indicator {
    position:absolute; top:1px; bottom:1px; width:18px;
    border-radius:50%; background:var(--yellow);
    box-shadow:0 0 10px var(--yellow); transition:left .3s ease;
    transform:translateX(-50%);
  }
  .lane-status { font-size:.75rem; text-align:center; margin-top:4px; }

  /* ── OBSTACLE ── */
  .obstacle-box {
    display:flex; align-items:center; gap:14px;
    background:rgba(0,0,0,.3); border-radius:10px;
    padding:12px; border:1px solid var(--border); margin-bottom:12px;
  }
  .obstacle-icon { font-size:2rem; }
  .obstacle-dist { font-family:'Orbitron',sans-serif; font-size:1.6rem; color:var(--cyan); }
  .obstacle-label { font-size:.7rem; color:var(--dim); }

  /* ── SIGNS ── */
  .sign-list { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; min-height:36px; }
  .sign-tag {
    background:rgba(255,23,68,.15); border:1px solid var(--red);
    color:var(--red); padding:4px 12px; border-radius:20px;
    font-size:.75rem; font-weight:600; letter-spacing:1px;
  }
  .sign-none { color:var(--dim); font-size:.8rem; align-self:center; }

  /* ── MINI METRICS ── */
  .metrics-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .metric { background:rgba(0,0,0,.3); border-radius:10px; padding:12px; border:1px solid var(--border); }
  .metric-val { font-family:'Orbitron',sans-serif; font-size:1.3rem; color:var(--cyan); }
  .metric-lbl { font-size:.68rem; color:var(--dim); margin-top:3px; text-transform:uppercase; letter-spacing:1px; }

  .gpad-btn {
    width:70px; height:60px; border-radius:10px;
    background:#0d1b2a; border:2px solid #1e3a5f;
    color:var(--text); font-size:1.1rem; cursor:pointer;
    transition:all .15s; display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:2px;
    user-select:none; -webkit-user-select:none;
  }
  .gpad-btn:active, .gpad-btn.pressed {
    background:var(--cyan); color:#000;
    border-color:var(--cyan);
    box-shadow:0 0 15px var(--cyan);
    transform:scale(.95);
  }

  /* ── FOOTER ── */
  footer {
    text-align:center; padding:14px; color:var(--dim);
    font-size:.75rem; border-top:1px solid var(--border);
  }

  @keyframes aeb-pulse {
    from { box-shadow: 0 0 10px #ff174488; }
    to   { box-shadow: 0 0 30px #ff1744cc, 0 0 60px #ff174466; }
  }

  /* ── KEYBOARD CONTROLS ── */
  .key-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .key-card {
    background:rgba(0,0,0,.35);
    border:1px solid var(--border);
    border-radius:12px;
    padding:10px 12px;
    display:flex;
    align-items:center;
    gap:10px;
    transition: all .3s;
  }
  .key-card.active {
    border-color: var(--active-col, var(--cyan));
    box-shadow: 0 0 14px var(--active-col, var(--cyan));
    background: rgba(0,229,255,.07);
  }
  .key-badge {
    min-width:36px; height:36px;
    border-radius:8px;
    border:2px solid var(--border);
    background:#050a12;
    display:flex; align-items:center; justify-content:center;
    font-family:'Orbitron',sans-serif;
    font-size:.85rem; font-weight:700;
    color:var(--dim);
    transition: all .3s;
    flex-shrink:0;
  }
  .key-card.active .key-badge {
    border-color: var(--active-col, var(--cyan));
    color: var(--active-col, var(--cyan));
    box-shadow: 0 0 10px var(--active-col, var(--cyan));
  }
  .key-info { flex:1; }
  .key-name {
    font-size:.72rem; font-weight:600; color:var(--text);
    letter-spacing:.5px; text-transform:uppercase;
  }
  .key-desc { font-size:.62rem; color:var(--dim); margin-top:2px; }
  .key-status {
    font-size:.65rem; font-weight:700;
    color:var(--dim); text-align:right;
    transition:color .3s;
  }
  .key-card.active .key-status { color:var(--active-col, var(--cyan)); }
</style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="logo">CARLA <span>AI</span> SYSTEM</div>
  <div class="live-badge"><div class="dot"></div> LIVE TELEMETRY</div>
  <div style="font-size:.8rem;color:var(--dim)" id="conn-status">Connecting...</div>
</header>

<!-- MAIN GRID -->
<div class="grid">

  <!-- LEFT: Traffic Light + Location -->
  <div>
    <div class="card" style="margin-bottom:18px">
      <div class="card-title">Traffic Signal</div>
      <div class="tl-wrap">
        <div class="tl-housing">
          <div class="bulb" id="b-red"></div>
          <div class="bulb" id="b-yellow"></div>
          <div class="bulb" id="b-green"></div>
        </div>
        <div class="tl-label">
          <div id="tl-text">---</div>
          <div style="font-size:.65rem;color:var(--dim);margin-top:4px">TRAFFIC LIGHT</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Location &amp; Heading</div>
      <div class="metrics-grid">
        <div class="metric">
          <div class="metric-val" id="loc-x">---</div>
          <div class="metric-lbl">X (m)</div>
        </div>
        <div class="metric">
          <div class="metric-val" id="loc-y">---</div>
          <div class="metric-lbl">Y (m)</div>
        </div>
        <div class="metric" style="grid-column:span 2">
          <div class="metric-val" id="heading">---</div>
          <div class="metric-lbl">Heading (deg)</div>
        </div>
      </div>
    </div>
  </div>

  <!-- CENTER -->
  <div>
    <!-- SPEED + MODE -->
    <div class="card" style="margin-bottom:18px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
          <div class="card-title">Speed</div>
          <div class="speed-wrap">
            <div class="speed-num" id="speed">0</div>
            <div class="speed-unit">km / h</div>
            <div class="speed-target">Target: <span id="target-speed">40</span> km/h</div>
          </div>
        </div>
        <div style="text-align:right">
          <div class="card-title">AI Mode</div>
          <div class="mode-badge" id="mode-badge">NORMAL</div>
          <div style="font-size:.75rem;color:var(--dim)">RL Epsilon: <span id="rl-eps">---</span></div>
        </div>
      </div>
    </div>

    <!-- ALERT BANNER (Hidden by default) -->
    <div id="alert-banner" style="display:none; margin-bottom:18px; padding:15px; background:rgba(255,0,255,0.2); border:2px solid #ff00ff; border-radius:12px; text-align:center; box-shadow:0 0 15px rgba(255,0,255,0.4);">
      <h2 id="alert-text" style="color:#ff00ff; margin:0; font-family:'Orbitron',sans-serif; text-shadow:0 0 10px #ff00ff;">!! YIELDING TO PEDESTRIAN !!</h2>
    </div>

    <!-- NAVIGATION DIRECTION -->
    <div class="card" style="margin-bottom:18px">
      <div class="card-title">&#9876; Next Turn Direction</div>
      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px">

        <!-- Big arrow display -->
        <div id="dir-arrow-box" style="
          width:80px;height:80px;border-radius:16px;
          background:#050a12;border:2px solid #1e3a5f;
          display:flex;flex-direction:column;
          align-items:center;justify-content:center;
          font-size:2.2rem;flex-shrink:0;
          transition:all .4s;
        ">&#8593;</div>

        <!-- Text info -->
        <div style="flex:1">
          <div id="dir-label" style="
            font-family:'Orbitron',sans-serif;
            font-size:1.3rem;font-weight:700;
            color:#00e676;letter-spacing:2px;
            transition:color .3s;
          ">FOLLOW</div>
          <div id="dir-detail" style="font-size:.72rem;color:var(--dim);margin-top:5px">
            Following current road
          </div>
        </div>

        <!-- Turn signal dots -->
        <div style="display:flex;flex-direction:column;gap:6px;align-items:center">
          <div id="sig-left"  style="width:14px;height:14px;border-radius:50%;background:#1e3a5f;transition:all .3s"></div>
          <div style="width:6px;height:6px;border-radius:50%;background:#333"></div>
          <div id="sig-right" style="width:14px;height:14px;border-radius:50%;background:#1e3a5f;transition:all .3s"></div>
        </div>

      </div>
    </div>

    <!-- CONTROLS -->
    <div class="card" style="margin-bottom:18px">
      <div class="card-title">Vehicle Controls</div>
      <div class="bar-row bar-throttle">
        <div class="bar-label"><span>THROTTLE</span><span id="throttle-val">0.00</span></div>
        <div class="bar-track"><div class="bar-fill" id="throttle-bar" style="width:0%"></div></div>
      </div>
      <div class="bar-row bar-brake">
        <div class="bar-label"><span>BRAKE</span><span id="brake-val">0.00</span></div>
        <div class="bar-track"><div class="bar-fill" id="brake-bar" style="width:0%"></div></div>
      </div>
      <div class="bar-row bar-steer">
        <div class="bar-label"><span>STEERING</span><span id="steer-val">0.00</span></div>
        <div class="bar-track"><div class="bar-fill" id="steer-bar" style="width:50%"></div></div>
      </div>
      <div class="metrics-grid" style="margin-top:14px">
        <div class="metric">
          <div class="metric-val" id="gear-val">N</div>
          <div class="metric-lbl">Gear</div>
        </div>
        <div class="metric">
          <div class="metric-val" id="accel-val">0.0</div>
          <div class="metric-lbl">Accel m/s²</div>
        </div>
      </div>
    </div>

    <!-- LANE -->
    <div class="card" style="margin-bottom:18px">
      <div class="card-title">Lane Detection</div>
      <div class="lane-track">
        <div class="lane-center-line"></div>
        <div class="lane-indicator" id="lane-ind" style="left:50%"></div>
      </div>
      <div class="lane-status" id="lane-status">Detecting...</div>
    </div>

    <!-- MANUAL CONTROL PANEL -->
    <div class="card" id="manual-panel">
      <div class="card-title">Manual Control</div>

      <!-- Toggle Button -->
      <button id="manual-toggle" onclick="toggleManual()" style="
        width:100%; padding:12px; border-radius:10px; border:2px solid #546e7a;
        background:#0d1b2a; color:#546e7a; font-family:'Orbitron',sans-serif;
        font-size:.85rem; letter-spacing:2px; cursor:pointer; margin-bottom:14px;
        transition:all .3s;
      ">ENABLE MANUAL MODE</button>

      <!-- WASD Gamepad -->
      <div id="gamepad" style="opacity:.4; pointer-events:none; transition:opacity .3s;">
        <!-- Row 1: W -->
        <div style="display:flex;justify-content:center;margin-bottom:4px">
          <button class="gpad-btn" id="btn-w"
            onmousedown="keyDown('w')" onmouseup="keyUp('w')"
            ontouchstart="keyDown('w')" ontouchend="keyUp('w')">&#9650;<br><span style='font-size:.6rem'>THROTTLE</span></button>
        </div>
        <!-- Row 2: A S D -->
        <div style="display:flex;justify-content:center;gap:4px;margin-bottom:4px">
          <button class="gpad-btn" id="btn-a"
            onmousedown="keyDown('a')" onmouseup="keyUp('a')"
            ontouchstart="keyDown('a')" ontouchend="keyUp('a')">&#9664;<br><span style='font-size:.6rem'>LEFT</span></button>
          <button class="gpad-btn" id="btn-s"
            onmousedown="keyDown('s')" onmouseup="keyUp('s')"
            ontouchstart="keyDown('s')" ontouchend="keyUp('s')">&#9660;<br><span style='font-size:.6rem'>BRAKE</span></button>
          <button class="gpad-btn" id="btn-d"
            onmousedown="keyDown('d')" onmouseup="keyUp('d')"
            ontouchstart="keyDown('d')" ontouchend="keyUp('d')">&#9654;<br><span style='font-size:.6rem'>RIGHT</span></button>
        </div>
        <!-- Row 3: Space + Reverse -->
        <div style="display:flex;gap:4px;margin-top:4px">
          <button class="gpad-btn" id="btn-space" style="flex:2;"
            onmousedown="keyDown('space')" onmouseup="keyUp('space')"
            ontouchstart="keyDown('space')" ontouchend="keyUp('space')">&#9646;&#9646; BRAKE</button>
          <button class="gpad-btn" id="btn-rev" style="flex:1;" onclick="toggleReverse()">
            &#8634; REV<br><span id="rev-status" style="font-size:.55rem">OFF</span>
          </button>
        </div>
      </div>
    </div>
  </div>

  <!-- RIGHT: Obstacle + Signs + Misc -->
  <div>
    <!-- AEB STATUS PANEL -->
    <div class="card" id="aeb-card" style="margin-bottom:18px;border-color:#1e3a5f;transition:all .3s">
      <div class="card-title">&#128721; Auto Emergency Braking (AEB)</div>
      <div style="display:flex;align-items:center;gap:14px">
        <!-- Status icon -->
        <div id="aeb-icon" style="
          width:56px;height:56px;border-radius:50%;
          background:#0a1520;border:3px solid #1e3a5f;
          display:flex;align-items:center;justify-content:center;
          font-size:1.6rem;flex-shrink:0;transition:all .3s;
        ">&#10003;</div>
        <!-- Text info -->
        <div style="flex:1">
          <div id="aeb-level-text" style="
            font-family:'Orbitron',sans-serif;font-size:1.1rem;
            font-weight:700;color:#00e676;letter-spacing:2px;
            transition:color .3s;
          ">CLEAR</div>
          <div id="aeb-detail" style="font-size:.72rem;color:var(--dim);margin-top:4px">
            No obstacle in braking zone
          </div>
        </div>
        <!-- Distance meter -->
        <div style="text-align:right">
          <div id="aeb-dist-val" style="
            font-family:'Orbitron',sans-serif;font-size:1.5rem;
            color:var(--dim);transition:color .3s;
          ">--</div>
          <div style="font-size:.6rem;color:var(--dim)">METRES</div>
        </div>
      </div>
      <!-- Proximity bar -->
      <div style="margin-top:12px">
        <div style="display:flex;justify-content:space-between;font-size:.65rem;color:var(--dim);margin-bottom:3px">
          <span>DANGER</span><span>SAFE</span>
        </div>
        <div style="height:8px;background:#0a1520;border-radius:4px;border:1px solid var(--border);overflow:hidden">
          <div id="aeb-prox-bar" style="height:100%;width:0%;border-radius:4px;transition:width .3s,background .3s;background:#00e676"></div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom:18px">
      <div class="card-title">Closest Obstacle (LiDAR)</div>
      <div class="obstacle-box" id="obstacle-box">
        <div class="obstacle-icon">🚗</div>
        <div>
          <div class="obstacle-dist"><span id="obs-dist">---</span> <span style="font-size:.9rem">m</span></div>
          <div class="obstacle-label">TTC: <span id="obs-ttc">---</span>s &nbsp;|&nbsp; <span id="obs-name">CLEAR</span></div>
        </div>
      </div>
    </div>


    <div class="card" style="margin-bottom:18px">
      <div class="card-title">YOLO Detected Signs</div>
      <div class="sign-list" id="sign-list">
        <span class="sign-none">No signs detected</span>
      </div>
    </div>

    <div class="card" style="margin-bottom:18px">
      <div class="card-title">Speed Limit Sign</div>
      <div style="text-align:center;padding:8px 0">
        <div style="display:inline-block;background:#0a1520;border:3px solid #ff6d00;
             border-radius:50%;width:80px;height:80px;line-height:80px;text-align:center;
             font-family:'Orbitron',sans-serif;font-size:1.4rem;font-weight:900;
             color:#ff6d00;box-shadow:0 0 20px #ff6d0055" id="speed-limit-badge">40</div>
        <div style="font-size:.7rem;color:var(--dim);margin-top:6px">km/h LIMIT</div>
      </div>
    </div>

    <div class="card" style="margin-bottom:18px">
      <div class="card-title">Traffic Behavior Prediction</div>
      <div id="traffic-pred-list" style="min-height:60px">
        <span style="color:var(--dim);font-size:.8rem">No vehicles tracked</span>
      </div>
    </div>

    <div class="card">
      <div class="card-title">System Status</div>
      <div class="metrics-grid">
        <div class="metric">
          <div class="metric-val" id="hand-brake">OFF</div>
          <div class="metric-lbl">Hand Brake</div>
        </div>
        <div class="metric">
          <div class="metric-val" id="reverse-val">NO</div>
          <div class="metric-lbl">Reverse</div>
        </div>
      </div>
    </div>

    <!-- ENVIRONMENT CONTROL PANEL -->
    <div class="card" style="margin-top:18px; text-align:center;">
      <div class="card-title">&#127782; Environment Control</div>
      <button onclick="changeWeather()" style="
        width:100%; padding:12px; border-radius:10px; border:2px solid #00e5ff;
        background:#050a12; color:#00e5ff; font-family:'Orbitron',sans-serif;
        font-size:.85rem; letter-spacing:2px; cursor:pointer;
        transition:all .3s;
      " onmouseover="this.style.background='#00e5ff'; this.style.color='#000'"
        onmouseout="this.style.background='#050a12'; this.style.color='#00e5ff'">
        CYCLE WEATHER
      </button>
    </div>

    <!-- KEYBOARD CONTROLS PANEL -->
    <div class="card" style="margin-top:18px">
      <div class="card-title">&#9000; Keyboard Controls</div>
      <div class="key-grid">

        <!-- P: Parking -->
        <div class="key-card" id="kc-p" style="--active-col:#ffea00">
          <div class="key-badge">P</div>
          <div class="key-info">
            <div class="key-name">Parking</div>
            <div class="key-desc">Auto park toggle</div>
          </div>
          <div class="key-status" id="ks-p">OFF</div>
        </div>

        <!-- R: RL Control -->
        <div class="key-card" id="kc-r" style="--active-col:#d500f9">
          <div class="key-badge">R</div>
          <div class="key-info">
            <div class="key-name">RL Mode</div>
            <div class="key-desc">Neural network</div>
          </div>
          <div class="key-status" id="ks-r">OFF</div>
        </div>

        <!-- H: Horn -->
        <div class="key-card" id="kc-h" style="--active-col:#00e5ff">
          <div class="key-badge">H</div>
          <div class="key-info">
            <div class="key-name">Horn</div>
            <div class="key-desc">Honk!</div>
          </div>
          <div class="key-status" id="ks-h">&#128508;</div>
        </div>

        <!-- D: Debug -->
        <div class="key-card" id="kc-d" style="--active-col:#00e676">
          <div class="key-badge">D</div>
          <div class="key-info">
            <div class="key-name">Debug</div>
            <div class="key-desc">Print location</div>
          </div>
          <div class="key-status" id="ks-d">&#128196;</div>
        </div>

        <!-- ESC: Exit -->
        <div class="key-card" id="kc-esc" style="--active-col:#ff1744;grid-column:span 2">
          <div class="key-badge" style="min-width:54px;font-size:.7rem">ESC</div>
          <div class="key-info">
            <div class="key-name">Exit Simulation</div>
            <div class="key-desc">Stops CARLA driving script</div>
          </div>
          <div class="key-status" id="ks-esc">&#9209;</div>
        </div>

      </div>

      <!-- Quick legend -->
      <div style="margin-top:12px;padding-top:10px;border-top:1px solid var(--border);display:flex;gap:8px;flex-wrap:wrap">
        <span style="font-size:.65rem;color:var(--dim)">MODE INDICATOR:</span>
        <span style="font-size:.65rem;color:#ffea00">&#9632; PARKING</span>
        <span style="font-size:.65rem;color:#d500f9">&#9632; RL ON</span>
        <span style="font-size:.65rem;color:var(--green)">&#9632; ACTIVE</span>
      </div>
    </div>

  </div>

</div><!-- /grid -->

<footer>
  CARLA Autonomous Driving System &nbsp;|&nbsp; Hackathon Demo &nbsp;|&nbsp;
  Updates every 300ms &nbsp;|&nbsp; <span id="last-update">--</span>
</footer>

<script>
const MODE_COLORS = {
  NORMAL:          {color:'#00e676', border:'#00e676'},
  BRAKING:         {color:'#ff6d00', border:'#ff6d00'},
  EMERGENCY:       {color:'#ff1744', border:'#ff1744'},
  PARKING:         {color:'#ffea00', border:'#ffea00'},
  RL_CONTROL:      {color:'#d500f9', border:'#d500f9'},
  STOPPED_AT_SIGNAL:{color:'#ff1744',border:'#ff1744'},
  AVOIDING:        {color:'#ff6d00', border:'#ff6d00'},
  MANUAL:          {color:'#ff9100', border:'#ff9100'},  // orange — manual mode
  YIELDING:        {color:'#ff00ff', border:'#ff00ff'},  // pink — yielding to ped
};

function setTrafficLight(state) {
  document.getElementById('b-red').className    = 'bulb' + (state==='RED'    ? ' on-red'    : '');
  document.getElementById('b-yellow').className = 'bulb' + (state==='YELLOW' ? ' on-yellow' : '');
  document.getElementById('b-green').className  = 'bulb' + (state==='GREEN'  ? ' on-green'  : '');
  const colors = {RED:'#ff1744',YELLOW:'#ffea00',GREEN:'#00e676',NONE:'#546e7a'};
  const el = document.getElementById('tl-text');
  el.textContent = state === 'NONE' ? 'NO LIGHT' : state;
  el.style.color = colors[state] || '#cfd8dc';
}

function setMode(mode) {
  const badge = document.getElementById('mode-badge');
  badge.textContent = mode;
  const c = MODE_COLORS[mode] || {color:'#00e5ff', border:'#00e5ff'};
  badge.style.color = c.color;
  badge.style.borderColor = c.border;
  badge.style.boxShadow = `0 0 12px ${c.color}55`;
  
  // Handle Alert Banner
  const alertBanner = document.getElementById('alert-banner');
  const alertText = document.getElementById('alert-text');
  
  if (mode === 'YIELDING') {
    alertBanner.style.display = 'block';
    alertBanner.style.borderColor = '#ff00ff';
    alertBanner.style.background = 'rgba(255,0,255,0.2)';
    alertBanner.style.boxShadow = '0 0 15px rgba(255,0,255,0.4)';
    alertBanner.style.animation = 'pulse 0.5s infinite alternate';
    alertText.style.color = '#ff00ff';
    alertText.style.textShadow = '0 0 10px #ff00ff';
    alertText.textContent = '!! YIELDING TO PEDESTRIAN !!';
  } else if (mode === 'STOPPED_AT_SIGNAL') {
    alertBanner.style.display = 'block';
    alertBanner.style.borderColor = '#ff1744';
    alertBanner.style.background = 'rgba(255,23,68,0.2)';
    alertBanner.style.boxShadow = '0 0 15px rgba(255,23,68,0.4)';
    alertBanner.style.animation = 'pulse 0.5s infinite alternate';
    alertText.style.color = '#ff1744';
    alertText.style.textShadow = '0 0 10px #ff1744';
    alertText.textContent = 'STOP SIGN - WAITING...';
  } else {
    alertBanner.style.display = 'none';
    alertBanner.style.animation = 'none';
  }
}

function setSigns(signs) {
  const el = document.getElementById('sign-list');
  if (!signs || signs.length === 0) {
    el.innerHTML = '<span class="sign-none">No signs detected</span>';
    return;
  }
  el.innerHTML = signs.map(s => `<span class="sign-tag">${s}</span>`).join('');
}

function setTrafficBehaviors(behaviors) {
  const el = document.getElementById('traffic-pred-list');
  if (!behaviors || Object.keys(behaviors).length === 0) {
    el.innerHTML = '<span style="color:var(--dim);font-size:.8rem">No vehicles tracked</span>';
    return;
  }
  
  let html = '<div style="display:flex;flex-direction:column;gap:6px">';
  for (const [beh, count] of Object.entries(behaviors)) {
    const colors = {
      "STATIONARY":   "#969696",
      "MOVING":       "#00e64c",
      "ACCELERATING": "#ffa500",
      "BRAKING":      "#ff1744",
      "LANE_CHANGE":  "#ff00d4"
    };
    const col = colors[beh] || "#646464";
    html += `<div style="display:flex;justify-content:space-between;font-size:.75rem;padding:4px 8px;background:rgba(0,0,0,0.2);border-radius:4px;border-left:3px solid ${col}">
               <span style="color:${col};font-weight:600">${beh}</span>
               <span style="color:#fff">${count}</span>
             </div>`;
  }
  html += '</div>';
  el.innerHTML = html;
}

function setDirection(dir) {
  const DIR_CFG = {
    'LEFT':     { arrow:'&#8592;', label:'TURN LEFT',  detail:'Turning left ahead',    color:'#00e5ff', sigL:true,  sigR:false },
    'RIGHT':    { arrow:'&#8594;', label:'TURN RIGHT', detail:'Turning right ahead',   color:'#00e5ff', sigL:false, sigR:true  },
    'STRAIGHT': { arrow:'&#8593;', label:'STRAIGHT',   detail:'Go straight at junction',color:'#00e676', sigL:false, sigR:false },
    'UTURN':    { arrow:'&#8635;', label:'U-TURN',     detail:'U-turn ahead',          color:'#ff6d00', sigL:true,  sigR:false },
    'FOLLOW':   { arrow:'&#8593;', label:'FOLLOW ROAD',detail:'Following current road', color:'#546e7a', sigL:false, sigR:false },
  };
  const cfg = DIR_CFG[dir] || DIR_CFG['FOLLOW'];

  const box    = document.getElementById('dir-arrow-box');
  const label  = document.getElementById('dir-label');
  const detail = document.getElementById('dir-detail');
  const sigL   = document.getElementById('sig-left');
  const sigR   = document.getElementById('sig-right');

  box.innerHTML       = cfg.arrow;
  box.style.color     = cfg.color;
  box.style.borderColor = cfg.color;
  box.style.boxShadow = dir !== 'FOLLOW' ? `0 0 18px ${cfg.color}66` : 'none';
  label.textContent   = cfg.label;
  label.style.color   = cfg.color;
  detail.textContent  = cfg.detail;

  // Turn signal indicators
  sigL.style.background = cfg.sigL ? cfg.color : '#1e3a5f';
  sigL.style.boxShadow  = cfg.sigL ? `0 0 10px ${cfg.color}` : 'none';
  sigR.style.background = cfg.sigR ? cfg.color : '#1e3a5f';
  sigR.style.boxShadow  = cfg.sigR ? `0 0 10px ${cfg.color}` : 'none';
}

function setAEB(level, dist, target) {

  const AEB_CFG = {
    'CLEAR':     { color:'#00e676', icon:'&#10003;',  label:'CLEAR',     detail:'No obstacle in braking zone',     border:'#1e3a5f' },
    'WARNING':   { color:'#ffea00', icon:'&#9888;',   label:'WARNING',   detail:`${target ? target.toUpperCase()+' ahead — ' : ''}Throttle cut`, border:'#ffea00' },
    'SOFT':      { color:'#ff6d00', icon:'&#128721;', label:'SOFT BRAKE',detail:`Braking for ${target || 'obstacle'}`, border:'#ff6d00' },
    'HARD':      { color:'#ff3030', icon:'&#128721;', label:'HARD BRAKE',detail:`${target ? target.toUpperCase()+' ' : ''}Emergency deceleration!`, border:'#ff3030' },
    'EMERGENCY': { color:'#ff1744', icon:'&#9888;',   label:'EMERGENCY', detail:`FULL STOP — ${target ? target.toUpperCase() : 'OBSTACLE'} DETECTED!`, border:'#ff1744' },
  };
  const cfg = AEB_CFG[level] || AEB_CFG['CLEAR'];

  const card     = document.getElementById('aeb-card');
  const icon     = document.getElementById('aeb-icon');
  const levelTxt = document.getElementById('aeb-level-text');
  const detail   = document.getElementById('aeb-detail');
  const distVal  = document.getElementById('aeb-dist-val');
  const proxBar  = document.getElementById('aeb-prox-bar');

  card.style.borderColor  = cfg.border;
  card.style.boxShadow    = level !== 'CLEAR' ? `0 0 18px ${cfg.color}44` : 'none';
  icon.innerHTML          = cfg.icon;
  icon.style.borderColor  = cfg.color;
  icon.style.color        = cfg.color;
  icon.style.boxShadow    = level !== 'CLEAR' ? `0 0 14px ${cfg.color}88` : 'none';
  levelTxt.textContent    = cfg.label;
  levelTxt.style.color    = cfg.color;
  detail.innerHTML        = cfg.detail;
  distVal.style.color     = cfg.color;

  if (dist < 999) {
    distVal.textContent = dist.toFixed(1);
    // Proximity bar: 0m=100%, 30m+=0%
    const pct = Math.max(0, Math.min(100, (1 - dist / 30) * 100));
    proxBar.style.width      = pct + '%';
    proxBar.style.background = cfg.color;
  } else {
    distVal.textContent      = '--';
    proxBar.style.width      = '0%';
    proxBar.style.background = '#00e676';
  }

  // Pulse animation for EMERGENCY
  if (level === 'EMERGENCY') {
    card.style.animation = 'aeb-pulse 0.4s infinite alternate';
  } else {
    card.style.animation = '';
  }
}

function setLane(center, detected) {
  const ind = document.getElementById('lane-ind');
  const pct = Math.max(5, Math.min(95, center * 100));
  ind.style.left = pct + '%';
  const dev = (center - 0.5).toFixed(2);
  const status = document.getElementById('lane-status');
  if (detected) {
    const devNum = Math.abs(center - 0.5);
    const col = devNum < 0.08 ? '#00e676' : devNum < 0.2 ? '#ffea00' : '#ff6d00';
    status.innerHTML = `<span style="color:${col}">LANE DETECTED &nbsp; Deviation: ${dev}</span>`;
    ind.style.background = col;
    ind.style.boxShadow = `0 0 10px ${col}`;
  } else {
    status.innerHTML = '<span style="color:#ff1744">LANE NOT DETECTED</span>';
    ind.style.background = '#546e7a';
    ind.style.boxShadow = 'none';
  }
}

function setObstacle(dist, ttc, name) {
  document.getElementById('obs-name').textContent = name || 'CLEAR';
  if (dist < 999) {
    document.getElementById('obs-dist').textContent = dist.toFixed(1);
    document.getElementById('obs-ttc').textContent  = ttc.toFixed(1);
    const box = document.getElementById('obstacle-box');
    const danger = ttc < 2.0;
    box.style.borderColor = danger ? '#ff1744' : ttc < 3.5 ? '#ff6d00' : '#1e3a5f';
    box.style.background  = danger ? 'rgba(255,23,68,.1)' : 'rgba(0,0,0,.3)';
  } else {
    document.getElementById('obs-dist').textContent = '--';
    document.getElementById('obs-ttc').textContent  = '--';
    document.getElementById('obs-name').textContent = 'CLEAR';
    document.getElementById('obstacle-box').style.borderColor = '#1e3a5f';
    document.getElementById('obstacle-box').style.background  = 'rgba(0,0,0,.3)';
  }
}

function updateDashboard() {
  fetch('/data')
    .then(r => r.json())
    .then(d => {
      if (d.error) {
        document.getElementById('conn-status').textContent = 'Error: ' + d.error;
        return;
      }

      document.getElementById('conn-status').textContent = 'Connected';
      document.getElementById('conn-status').style.color = '#00e676';

      // Speed
      const spEl = document.getElementById('speed');
      spEl.textContent = d.speed.toFixed(0);
      const tgt = (d.ai && d.ai.speed) ? 40 : 40;
      spEl.style.color = d.speed < 5 ? '#ff6d00' : '#00e676';

      // Controls
      document.getElementById('throttle-val').textContent = d.throttle.toFixed(2);
      document.getElementById('brake-val').textContent    = d.brake.toFixed(2);
      document.getElementById('steer-val').textContent    = d.steer.toFixed(2);
      document.getElementById('throttle-bar').style.width = (d.throttle * 100) + '%';
      document.getElementById('brake-bar').style.width    = (d.brake * 100) + '%';
      // steer bar: -1 to +1 → 0% to 100%, center=50%
      document.getElementById('steer-bar').style.width    = (50 + d.steer * 50) + '%';
      document.getElementById('accel-val').textContent    = d.acceleration;
      document.getElementById('hand-brake').textContent   = d.hand_brake ? 'ON' : 'OFF';
      document.getElementById('reverse-val').textContent  = d.reverse ? 'YES' : 'NO';
      let g = d.gear; if (g===-1) g='R'; else if (g===0) g='N';
      document.getElementById('gear-val').textContent = g;

      // Location
      document.getElementById('loc-x').textContent    = d.location.x;
      document.getElementById('loc-y').textContent    = d.location.y;
      document.getElementById('heading').textContent  = d.heading + '°';

      // Traffic light
      setTrafficLight(d.traffic_light);

      // AI telemetry
      if (d.ai && Object.keys(d.ai).length > 0) {
        setMode(d.ai.mode || 'NORMAL');
        setSigns(d.ai.detected_signs);
        const ttc  = d.ai.ttc  || 999;
        const dist = ttc < 999 ? ttc * Math.max(d.speed / 3.6, 0.1) : 999;
        setObstacle(dist, ttc, d.ai.closest_obstacle);
        setLane(d.ai.lane_center || 0.5, d.ai.lane_detected || false);
        document.getElementById('rl-eps').textContent = '---';

        // NEW: Speed Limit badge
        const sl = d.ai.speed_limit || 40;
        const badge = document.getElementById('speed-limit-badge');
        badge.textContent = sl;
        const slColor = sl < 40 ? '#ff1744' : sl < 60 ? '#ff6d00' : '#00e676';
        badge.style.borderColor = slColor;
        badge.style.color = slColor;
        badge.style.boxShadow = `0 0 20px ${slColor}55`;

        // NEW: Traffic Behavior Prediction panel
        const behaviors = d.ai.traffic_behaviors || {};
        setTrafficBehaviors(behaviors);

        // AEB status panel
        setAEB(d.ai.aeb_level || 'CLEAR', d.ai.aeb_dist || 999, d.ai.aeb_target || '');

        // Navigation direction
        setDirection(d.ai.next_turn || 'FOLLOW');

        // NEW: Keyboard controls live highlight
        updateKeyControls(d.ai.mode || 'NORMAL');
      }

      document.getElementById('last-update').textContent =
        new Date().toLocaleTimeString();
    })
    .catch(() => {
      document.getElementById('conn-status').textContent = 'Disconnected';
      document.getElementById('conn-status').style.color = '#ff1744';
    });
}

setInterval(updateDashboard, 300);
updateDashboard();

// ── KEYBOARD CONTROLS LIVE HIGHLIGHTER ────────────────────────────
function updateKeyControls(mode) {
  // P key — active when PARKING mode
  const pActive = (mode === 'PARKING');
  const pCard   = document.getElementById('kc-p');
  const pStat   = document.getElementById('ks-p');
  if (pActive) { pCard.classList.add('active'); pStat.textContent = 'ON &#128663;'; }
  else         { pCard.classList.remove('active'); pStat.textContent = 'OFF'; }

  // R key — active when RL_CONTROL mode
  const rActive = (mode === 'RL_CONTROL');
  const rCard   = document.getElementById('kc-r');
  const rStat   = document.getElementById('ks-r');
  if (rActive) { rCard.classList.add('active'); rStat.textContent = 'ON &#129504;'; }
  else         { rCard.classList.remove('active'); rStat.textContent = 'OFF'; }
}

// ── MANUAL CONTROL ─────────────────────────────────────────────────
let isManual = false;
let reverseOn = false;  // Toggle state for reverse
const keys = {w:false, a:false, s:false, d:false, space:false};

function toggleReverse() {
  if (!isManual) return;
  reverseOn = !reverseOn;
  const btn = document.getElementById('btn-rev');
  const lbl = document.getElementById('rev-status');
  if (reverseOn) {
    btn.style.background = '#ff1744';
    btn.style.borderColor = '#ff1744';
    btn.style.color = '#fff';
    btn.style.boxShadow = '0 0 15px #ff174488';
    lbl.textContent = 'ON';
  } else {
    btn.style.background = '';
    btn.style.borderColor = '';
    btn.style.color = '';
    btn.style.boxShadow = '';
    lbl.textContent = 'OFF';
  }
}

function toggleManual() {
  isManual = !isManual;
  const btn = document.getElementById('manual-toggle');
  const pad = document.getElementById('gamepad');
  const panel = document.getElementById('manual-panel');

  if (isManual) {
    btn.textContent = 'DISABLE MANUAL MODE';
    btn.style.borderColor = '#ff1744';
    btn.style.color = '#ff1744';
    btn.style.boxShadow = '0 0 16px #ff174488';
    panel.style.borderColor = '#ff1744';
    pad.style.opacity = '1';
    pad.style.pointerEvents = 'auto';
  } else {
    btn.textContent = 'ENABLE MANUAL MODE';
    btn.style.borderColor = '#546e7a';
    btn.style.color = '#546e7a';
    btn.style.boxShadow = 'none';
    panel.style.borderColor = '#1e3a5f';
    pad.style.opacity = '.4';
    pad.style.pointerEvents = 'none';
    // Release all controls
    Object.keys(keys).forEach(k => keys[k] = false);
    sendControl();
  }
  fetch('/control', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({manual_mode: isManual})
  });
  if (!isManual) {
    // Reset reverse too
    reverseOn = false;
    const btn = document.getElementById('btn-rev');
    if (btn) { btn.style.background=''; btn.style.borderColor=''; btn.style.color=''; btn.style.boxShadow=''; }
    const lbl = document.getElementById('rev-status');
    if (lbl) lbl.textContent = 'OFF';
  }
}

function keyDown(k) {
  if (!isManual) return;
  keys[k] = true;
  const el = document.getElementById('btn-'+k);
  if (el) el.classList.add('pressed');
}
function keyUp(k) {
  keys[k] = false;
  const el = document.getElementById('btn-'+k);
  if (el) el.classList.remove('pressed');
}

// Keyboard support (browser dashboard)
// W=Throttle  A=Left  S=Brake  D=Right  Space=Hard Brake  X=Reverse Toggle
document.addEventListener('keydown', e => {
  if (!isManual) return;
  const map = {ArrowUp:'w', KeyW:'w', ArrowDown:'s', KeyS:'s',
                ArrowLeft:'a', KeyA:'a', ArrowRight:'d', KeyD:'d',
                Space:'space'};
  const k = map[e.code];
  if (k) { e.preventDefault(); keyDown(k); }

  // X key = Reverse toggle
  if (e.code === 'KeyX') {
    e.preventDefault();
    toggleReverse();
  }
});
document.addEventListener('keyup', e => {
  const map = {ArrowUp:'w', KeyW:'w', ArrowDown:'s', KeyS:'s',
                ArrowLeft:'a', KeyA:'a', ArrowRight:'d', KeyD:'d',
                Space:'space'};
  const k = map[e.code];
  if (k) keyUp(k);
});

function changeWeather() {
  fetch('/weather', { method: 'POST' }).catch(()=>{});
}

function sendControl() {
  if (!isManual) return;

  let throttle, brake;
  if (reverseOn) {
    // Reverse mode: W = go back (throttle), S = brake
    throttle = keys.w ? 0.7 : 0.0;
    brake    = (keys.s || keys.space) ? (keys.space ? 1.0 : 0.6) : 0.0;
  } else {
    // Forward mode: W = go forward, S = brake
    throttle = keys.w ? 0.7 : 0.0;
    brake    = (keys.s || keys.space) ? (keys.space ? 1.0 : 0.6) : 0.0;
  }
  const steer   = keys.a ? -0.5 : keys.d ? 0.5 : 0.0;
  const reverse = reverseOn;

  fetch('/control', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({manual_mode:true, throttle, steer, brake, reverse})
  }).catch(()=>{});
}

// Send control at 10Hz when manual is active
setInterval(() => { if (isManual) sendControl(); }, 100);
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  CARLA AI DASHBOARD SERVER")
    print("=" * 60)
    t = threading.Thread(target=connect_to_carla, daemon=True)
    t.start()
    print("[OK] Open in browser: http://localhost:5000")
    print("[OK] Keep CARLA + advanced_drive.py running")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)