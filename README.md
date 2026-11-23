# Farm Security – Person tracking + Number plate watch

Starter setup for one **Reolink RLC-823S2** PTZ camera.

Everything is **free and open source**, and runs on site with no cloud and no subscriptions:

| Part | Job | Licence |
|---|---|---|
| [Frigate](https://github.com/blakeblackshear/frigate) | records, detects people & vehicles, draws boxes, PTZ control | MIT |
| [FastALPR](https://github.com/ankandrew/fast-alpr) (inside plate-watcher) | finds + reads number plates from full 4K frames | MIT |
| plate-watcher (our code) | checks plates against `whitelist.yaml`, logs every vehicle, alerts on unknowns | ours |
| [ntfy](https://github.com/binwiederhier/ntfy) | push alerts with photo to phones | Apache-2.0 / GPL-2.0 |
| Mosquitto | message bus between them | EPL/EDL |

How a plate gets read: Frigate spots a moving car on the low-res stream → plate-watcher grabs a **4K frame**, crops round the car, FastALPR finds and reads the plate → it's checked against the whitelist (allowing 1 misread character) → unknown = alert. Frigate's own built-in plate reader also feeds in as a second opinion.

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
cd farm-security
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

- **Live view** with people/vehicles detected.
- **Debug view** (camera → ⋮ → Debug → *Bounding boxes*) shows the boxes round each person/vehicle live.
- **Review** tab – every alert/detection with clip and snapshot (boxes drawn on).
- **Explore** – search past objects; recognised plates show as labels on cars.

## 5. Whitelist (owner's cars)

Edit `plate-watcher/whitelist.yaml`:

```yaml
vehicles:
  - plate: AB12 CDE
    owner: Owner – Land Rover
  - plate: YX19 ABC
    owner: Haulier – J Smith
```

Spaces/case don't matter. One misread character (e.g. B read as 8) still counts as a match. Changes load automatically.

Watch it working:

```bash
docker logs -f plate-watcher
```

## 6. Alerts on your phone (ntfy – free, open source)

1. Set `HOST_IP` (this PC's IP) and `NTFY_TOPIC` (any hard-to-guess name) in `.env`, then `docker compose up -d`.
2. Install the **ntfy** app (Android Play Store / F-Droid, or iPhone App Store).
3. In the app: **+** → *Use another server* → `http://HOST_IP:8080` → topic = your `NTFY_TOPIC`.
4. Test: `curl -d "test alert" http://HOST_IP:8080/farm-alerts` – your phone should buzz.

Unknown plate → photo + plate + time. Same plate won't re-alert for 10 minutes.
Phones get alerts while on the site Wi-Fi/network. For alerts when away, use a free open-source VPN such as WireGuard back to the site (ask and I'll set it up).

*Optional:* Telegram also works (free but not open source) – fill in `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` (create a bot via @BotFather).

## 7. Tuning knobs

`plate-watcher` (set under `environment:` in `docker-compose.yml`):

| Setting | Default | Meaning |
|---|---|---|
| `MIN_SCORE` | 0.7 | ignore plate reads less confident than this |
| `MATCH_DISTANCE` | 1 | characters allowed to differ and still match whitelist |
| `DECISION_DELAY` | 20 | seconds to collect reads before deciding |
| `ALERT_COOLDOWN` | 600 | seconds before re-alerting the same plate |
| `GRAB_INTERVAL` / `MAX_GRABS` | 2 / 5 | grab a 4K frame every 2 s, up to 5 per vehicle |
| `ALPR_OCR_MODEL` | cct-s-v2-global-model | plate OCR model (global, recognises UK plates) |
| `OWN_ALPR` | 1 | 0 = only use Frigate's built-in plate reader |

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

**Better free person model (later):** Frigate's bundled model is basic. Once running, it can be swapped for a free YOLOv9 or D-FINE model (open-source weights, exported to ONNX) – see Frigate docs → *Object Detectors*. Best done on the Intel mini PC.

## 8. Known limits / next steps

- **Night plates**: IR makes reflective UK plates glare white. Expect daytime plates to work first; night needs spotlight/shutter tuning or a dedicated fixed plate camera at the gate.
- **Numbered boxes**: Frigate draws boxes + labels; a live view with a number per person is a later add-on.
- **Zones** (gate, yard, lairage) – add once the site layout is known, so alerts only fire where it matters.
- **Legal (UK GDPR)**: before going live – CCTV/ANPR signage, DPIA for ANPR, ICO registration, retention policy, tell staff. Default retention here: recordings 3 days, events/snapshots 30 days.

## Troubleshooting

| Problem | Fix |
|---|---|
| `check_camera.sh` port 554 closed | RTSP not enabled on camera |
| Stream FAIL / 401 | wrong user/password, or special characters in password |
| Frigate shows no video | `docker logs frigate`; check `.env` values |
| Plates never read | car too small in frame – zoom in more on Preset 1; check Frigate *Explore* for plate labels |
| No phone alert | `docker logs plate-watcher`; check the ntfy app server address + topic match `.env` |
| "Could not grab frame" in logs | Frigate restream not running – check Frigate is up and the camera name in the config |
