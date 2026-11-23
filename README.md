# Farm Security – Person tracking + Number plate watch

Starter setup for one **Reolink RLC-823S2** PTZ camera.

Everything is **free and open source**, and runs on site with no cloud and no subscriptions:

| Part | Job | Licence |
|---|---|---|
| **vision** (our code) | ONE live view that does **both at once**: numbered boxes round people (Person 1, 2…) **and** boxes round vehicles with their number plate + KNOWN / UNKNOWN. Sends alerts. | ours |
| ├ [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | finds people **and** vehicles in one pass | Apache-2.0 |
| ├ [Norfair](https://github.com/tryolabs/norfair) | follows each person/vehicle between frames so they keep their number | BSD-3 |
| ├ [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) | body pose (17 points) → what each person is doing | Apache-2.0 |
| └ [FastALPR](https://github.com/ankandrew/fast-alpr) | finds + reads the number plate from a 4K crop of each vehicle | MIT |
| [Frigate](https://github.com/blakeblackshear/frigate) | records 24/7, review/playback of events, PTZ control | MIT |
| [ntfy](https://github.com/binwiederhier/ntfy) | push alerts with photo to phones | Apache-2.0 / GPL-2.0 |
| [Mosquitto](https://github.com/eclipse-mosquitto/mosquitto) | message bus Frigate uses | EPL/EDL |

How it works, every frame (5 per second):

```
camera 4K stream ─► YOLOX finds people + vehicles ─► Norfair tracks them
      │                         people ──► "Person 1, 2, 3…" boxes  ──► person alert
      │                         vehicles ─► "Vehicle 1" box
      │                              └─► 4K crop ─► FastALPR reads plate ─► whitelist check
      │                                   ─► green "OK owner"  or  red "UNKNOWN!" + alert
      └─► live view: full-quality picture at 1080p with all the boxes drawn on
```

---

## 1. What you need

| Item | Notes |
|---|---|
| RLC-823S2 camera | |
| PoE+ switch (802.3at) or PoE+ injector | the PTZ motors + spotlight need PoE+ |
| A PC to run it | To start: any PC/laptop with Linux. Recommended: Intel 12th-gen+ mini PC (e.g. N100/i5), 16 GB RAM, 1–2 TB disk |
| Network cable | camera ↔ switch ↔ router |

## 2. Set up the camera (bench test first – on a desk, not up a pole)

1. Plug the camera into the PoE+ switch. Wait ~1 min.
2. Open the **Reolink app** (phone) or **Reolink Client** (PC), find the camera, set a **strong admin password**.
   *Don't use `@ : / ? #` in the password – they break video links.*
3. **Update firmware** (Settings → Device info / Firmware).
4. **Network → Advanced → Server settings / Port settings**
   - Enable **RTSP** (port 554)
   - Enable **ONVIF** (port 8000)
   - Disable **UID / P2P** (stops it talking to Reolink's cloud). Ideally also block the camera from the internet on your router.
5. Give it a **fixed IP** (static IP in the camera, or a DHCP reservation in the router). Write it down.
6. **System → Date & time**: Europe/London, NTP on.
7. **Display/Stream**: main stream 3840×2160, H.265, ~15 fps. Leave sub stream on default.
8. **Later, once mounted** (PTZ):
   - Point it at the gate/entrance, zoomed so a number plate fills a good chunk of the picture → save as **Preset 1**.
   - Turn on **Auto-tracking** for people and set it to **return to the guard position (Preset 1)** after a short time, so it's back watching the gate for cars.

## 3. Set up the PC

Ubuntu 24.04 is easiest. Install Docker:

```bash
curl -fsSL https://get.docker.com | sudo sh
```

```bash
sudo usermod -aG docker $USER   # then log out and back in
```

Copy this folder onto the PC, then:

```bash
cd farm
```

```bash
cp .env.example .env
```

Edit `.env` → put in the camera IP, username, password.

**Test the camera connection:**

```bash
./scripts/check_camera.sh
```

You should see both streams with codec + resolution. If the sub stream isn't 640×360, update `detect: width/height` in `frigate/config/config.yml`.

**Start everything:**

```bash
docker compose up -d --build
```

Get Frigate's first-login password:

```bash
docker logs frigate 2>&1 | grep -i password
```

Open **https://PC-IP:8971** (accept the certificate warning), log in as `admin`, change the password.

## 4. What you'll see

- **Frigate** (https://localhost:8971): recordings, live camera, PTZ arrows.
- **Debug view** (camera → ⋮ → Debug → *Bounding boxes*) shows the boxes round each person/vehicle live.
- **Review** tab – every alert/detection with clip and snapshot (boxes drawn on).
- **Explore** – search past objects.

## 5. The dashboard – http://localhost:8090

One page with everything (works on a phone too: `http://HOST_IP:8090` on the farm Wi-Fi):

| Section | What it shows |
|---|---|
| **Live view** | Full-quality camera picture with numbered boxes: **Person 1 90%**, **Vehicle 2 93% KL70XYZ UNKNOWN!** |
| **In view now** | A card for each person/vehicle with its confidence %, plate, country, region, plate date, colour |
| **Activity log** | Every person and vehicle seen: time, confidence %, plate details, allowed/unknown, ▶ play the clip |
| **Vehicles & plates** | Every plate read: country, region registered, when the plate was issued, type, approx colour, vehicle % and plate % |
| **Recordings** | Clips **with the boxes drawn on**; click to play. Each also has the original 4K (no boxes) |
| **Phone alerts** | QR code + steps to get alerts on your phone, **Send test alert** button, list of sent alerts (✓ delivered / ✗ failed) |

Header: camera fps, delay behind live, ● REC while recording, today's totals.

**What the plate tells us** (worked out locally, no internet): current UK plates (2001+) like `YX19 ABC` →
first letter = region first registered (Y = Yorkshire), numbers = when the plate was issued (19 = Mar–Aug 2019,
69 = Sep 2019–Feb 2020). Older UK formats, Northern Ireland and foreign plates are recognised too.
Private plates move between cars, so the date belongs to the plate. Colour is estimated from the picture.

**Recordings**: start when a person/vehicle appears, stop 10 s after they leave. Two files per clip in
`recordings/YYYY-MM-DD/`: `HH-MM-SS.mp4` (with boxes, 1080p) and `HH-MM-SS_4k.mp4` (original camera video).
Kept 30 days (`RECORD_KEEP_DAYS`). Allow roughly 3–5 GB per hour of activity.

### What people are doing

Each person's label shows what they're doing, e.g. **Person 3 90% walking** or **Person 2 crouching**, with a skeleton drawn on.

| Behaviour | How it's worked out | Alert by default |
|---|---|---|
| standing still / walking / running | how fast their feet move (in body-heights per second), with camera panning subtracted | – (logged) |
| crouching | hips close to ankles (body pose) | – (logged) |
| **lying down (possible fall)** | body horizontal for 4+ s | ✅ |
| **climbing** | one leg raised high + a hand above the shoulder for 2+ s | ✅ |
| arms raised | both hands above the head | – (logged) |
| **loitering** | in view for 2+ minutes (`LOITER_SECONDS`) | ✅ |
| **entered a restricted zone** | feet inside a zone you drew (Zones tab) | ✅ |

Everything appears in the **Activity log** (e.g. *Person 3 → lying down (possible fall)*) and in each person's row.
Choose which ones alert with `BEHAVIOUR_ALERTS` in `docker-compose.yml`.

How reliable is it? Movement, loitering and zones are reliable. Pose-based ones (crouching, lying, climbing) are good
when the person is reasonably big in the picture and not cut off by the edge; they get less reliable far away or at night.
People cut off by the picture edge are never called crouching/lying (missing legs would look like it).
A person already lying down when they first come into view may not be detected at all – falls are caught best when
the camera sees them standing first.

### Zones

**Zones tab** → click corners on the camera picture → name it → *Restricted* (alert) or *Watch* (log only) → optional
hours like `18:00-07:00` → Save. Works straight away. Saved in `settings/zones.yaml`.
Zones are fixed to the picture, so draw them with the PTZ at its home/gate position.

### Owner's cars (whitelist)

Edit `settings/whitelist.yaml` (`open -e settings/whitelist.yaml` on Mac):

```yaml
vehicles:
  - plate: AB12 CDE
    owner: Owner – Land Rover
  - plate: YX19 ABC
    owner: Haulier – J Smith
```

Spaces/case don't matter. One misread character (e.g. B read as 8) still counts as a match. Changes load automatically – no restart.

## 6. Alerts on your phone (ntfy – free, open source)

Open the dashboard → **Phone alerts** tab and follow the steps there:

1. Install the free **ntfy** app (iPhone App Store / Google Play).
2. Tap **+**, type the topic name shown on the dashboard, leave *Use another server* **unticked**, Subscribe.
3. Press **Send test alert** on the dashboard – your phone pings.

By default alerts go through the free public **ntfy.sh** server, so they reach your phone anywhere (4G, home, etc.).
A private random topic name is generated for you (saved in `data/ntfy_topic.txt`) so strangers can't read your alerts.
Note: alert photos/plates pass through ntfy.sh. For a fully private setup set `NTFY_SERVER=http://ntfy` and
`NTFY_PUBLIC_URL=http://HOST_IP:8080` in `.env` – then phones only get alerts on the farm Wi-Fi (or over a VPN).

**🔔 Person alerts ON / 🔕 OFF** – the button at the top of the dashboard. It's remembered across restarts.
Number-plate alerts are always on.

While **ON**, your phone gets:
- **Person seen: Person 3 - walking** – sent ~2 s after someone appears (so it can say what they're doing), with the
  picture showing the numbered boxes. People arriving together share one alert; at most one every 20 s.
- **Person 3 running / crouching / arms raised / climbing / lying down / loitering / entered <zone>** – whenever someone
  starts doing something (each thing once a minute per person). Falls, climbing and restricted zones are marked ⚠ urgent.
- **Video** – when the recording ends, the clip **with the boxes drawn on** is sent too (made smaller if needed to fit
  the 15 MB limit; long clips: first 90 s).

Always (whatever the button):
- **UNKNOWN VEHICLE: KL70XYZ** – photo of the vehicle with the **number plate enlarged**, plus country, region,
  plate date and colour – then the **video** when the clip ends. Same plate re-alerts at most once an hour.

While OFF, everything is still logged and recorded – your phone just isn't pinged for people.
`PERSON_ALERT_HOURS=18:00-07:00` in `.env` limits person alerts (while ON) to those hours.
Free ntfy.sh limits: 15 MB per file, attachments kept for a few hours – save any clip you need to keep
(the full recordings are always on the computer).

## 7. Tuning knobs

`vision` settings (under `environment:` in `docker-compose.yml`, then `docker compose up -d`):

| Setting | Default | Meaning |
|---|---|---|
| `FPS` | 5 | detections per second – lower if the computer struggles |
| `DISPLAY_WIDTH` | 1920 | live view size: 1920 (1080p), 2560 (1440p), 3840 (full 4K) |
| `DISPLAY_FPS` | 12 | live view refresh rate |
| `MIN_SCORE` | 0.45 | how sure the detector must be that it's a person/vehicle |
| `PLATE_MIN_SCORE` | 0.7 | how sure a plate read must be before deciding known/unknown |
| `PLATE_ALERT_COOLDOWN` | 3600 | seconds before re-alerting the same unknown plate |
| `PERSON_ALERT_HOURS` | always | while the button is ON: `always` or e.g. `18:00-07:00` |
| `PERSON_ALERT_GAP` | 20 | min seconds between "person seen" alerts |
| `SEND_CLIPS` | 1 | 0 = don't send video clips to the phone |
| `ALPR_OCR_MODEL` | cct-s-v2-global-model | plate OCR model (global, recognises UK plates) |
| `PLATES` | 1 | 0 = turn plate reading off |
| `POSE` | 1 | 0 = turn body pose off (movement, loitering and zones still work) |
| `LOITER_SECONDS` | 120 | in view this long = loitering |
| `BEHAVIOUR_ALERTS` | lying down,climbing,restricted zone,loitering | which behaviours send phone alerts (also: running, crouching, arms raised) |
| `RECORD` / `RECORD_KEEP_RAW` | 1 / 1 | turn recording off / don't keep the original 4K copy |
| `RECORD_KEEP_DAYS` | 30 | delete recordings older than this |

**Faster detection on Intel mini PC:** uncomment the `devices:` lines in `docker-compose.yml`, uncomment `hwaccel_args: preset-vaapi`, and replace the `detectors:` block in the Frigate config with:

```yaml
detectors:
  ov:
    type: openvino
    device: GPU
model:
  width: 300
  height: 300
  input_tensor: nhwc
  input_pixel_format: bgr
  path: /openvino-model/ssdlite_mobilenet_v2.xml
  labelmap_path: /openvino-model/coco_91cl_bkgr.txt
```

**Frigate's own detection** (used for its recordings/review) uses a basic bundled model; the live view and alerts come from `vision` (YOLOX). Frigate's model can later be swapped for a free YOLOv9 – see Frigate docs → *Object Detectors*.

## 8. Known limits / next steps

- **Night plates**: IR makes reflective UK plates glare white. Expect daytime plates to work first; night needs spotlight/shutter tuning or a dedicated fixed plate camera at the gate.
- **Parked cars**: a parked car is read a few times then left alone; if the PTZ moves, vehicles get re-read (unknown plates still only alert once an hour).
- **People numbers** are per sighting: if someone leaves view for more than ~3 s and comes back, they get a new number (no face recognition, on purpose).
- **Zones** (gate, yard, lairage) – add once the site layout is known, so alerts only fire where it matters.
- **Legal (UK GDPR)**: before going live – CCTV/ANPR signage, DPIA for ANPR, ICO registration, retention policy, tell staff. Default retention here: recordings 3 days, events/snapshots 30 days.

## Troubleshooting

| Problem | Fix |
|---|---|
| `check_camera.sh` port 554 closed | RTSP not enabled on camera |
| Stream FAIL / 401 | wrong user/password, or special characters in password |
| Frigate shows no video | `docker logs frigate`; check `.env` values |
| Plates never read | plate too small – zoom in so it's ~150 px wide in the 4K image; check `docker logs vision` for `read` lines |
| No phone alert | `docker logs vision`; check the ntfy app server address + topic match `.env` |
| "Cannot open" / "Could not grab" in vision logs | Frigate not running yet – `docker compose ps`; it retries automatically |
| Header shows "delay" above 1 s, or logs say "Fell behind live" | the computer can't decode 4K fast enough – see next row |
| Live view slow / lagging behind / Mac hot | lower `FPS` to 3 and `DISPLAY_FPS` to 8; in the Reolink app set the main stream to 15 fps (and H.264 if offered – easier to decode); Docker Desktop → Settings → Resources → give it more CPUs. `docker logs vision` shows "Stream 3840x2160 at N fps" – N should be close to the camera's frame rate |
| Camera clock in the picture doesn't match real time | Reolink app → Settings → Date & Time → turn on NTP / sync time |
| Frigate live view blurry | click the camera → stream menu → choose **Main (4K)** |
