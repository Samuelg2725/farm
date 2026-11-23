# Farm Security – Person tracking + Number plate watch

Starter setup for one **Reolink RLC-823S2** PTZ camera.

Everything is **free and open source**, and runs on site with no cloud and no subscriptions:

| Part | Job | Licence |
|---|---|---|
| **vision** (our code) | ONE live view that does **both at once**: numbered boxes round people (Person 1, 2…) **and** boxes round vehicles with their number plate + KNOWN / UNKNOWN. Sends alerts. | ours |
| ├ [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | finds people **and** vehicles in one pass | Apache-2.0 |
| ├ [Norfair](https://github.com/tryolabs/norfair) | follows each person/vehicle between frames so they keep their number | BSD-3 |
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

## 5. The live view (people + plates together)

Open **http://localhost:8090** (or `http://HOST_IP:8090` from a phone/laptop on the network).

- **People**: coloured box labelled **Person 1, Person 2…** – same person keeps their number while in view. Numbers restart each day.
- **Vehicles**: box labelled **Vehicle 1** plus the plate once read:
  - yellow `checking...` – plate read, confirming
  - green `OK <owner>` – on the whitelist
  - red `UNKNOWN!` – not on the whitelist → alert sent
  - grey – no plate visible (e.g. side-on)
- Top bar: time, number of people and vehicles in view. Lists underneath the picture.
- Everything is logged to `data/vision.db` (tables `people` and `vehicles`).

Watch what it's doing:

```bash
docker logs -f vision
```

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

1. Set `HOST_IP` (this computer's IP) and `NTFY_TOPIC` (any hard-to-guess name) in `.env`, then `docker compose up -d`.
2. Install the **ntfy** app (Android Play Store / F-Droid, or iPhone App Store).
3. In the app: **+** → *Use another server* → `http://HOST_IP:8080` → topic = your `NTFY_TOPIC`.
4. Test: `curl -d "test alert" http://HOST_IP:8080/farm-alerts` – your phone should buzz.

You get:
- **UNKNOWN VEHICLE: <plate>** – with a 4K photo of the vehicle. Same plate won't re-alert for an hour.
- **Person seen: Person 3** – with the numbered live picture. Choose when in `.env`: `PERSON_ALERT_HOURS=always`, `never`, or e.g. `18:00-07:00` (out of hours only). Max one per 5 minutes.

Phones get alerts while on the site network. For alerts when away, use a free open-source VPN such as WireGuard back to the site.

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
| `PERSON_ALERT_HOURS` | always | `always` / `never` / `18:00-07:00` |
| `PERSON_ALERT_COOLDOWN` | 300 | seconds between person alerts |
| `ALPR_OCR_MODEL` | cct-s-v2-global-model | plate OCR model (global, recognises UK plates) |
| `PLATES` | 1 | 0 = turn plate reading off |

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
| Live view slow / lagging behind / Mac hot | lower `FPS` to 3 and `DISPLAY_FPS` to 8; in the Reolink app set the main stream to 15 fps. `docker logs vision` shows "Stream 3840x2160 at N fps" – N should be close to the camera's frame rate |
| Frigate live view blurry | click the camera → stream menu → choose **Main (4K)** |
