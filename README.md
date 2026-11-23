# Farm Security

On-premises video analytics for a single PTZ camera (Reolink RLC-823S2). The system detects and tracks
people and vehicles, reads number plates against a list of authorised vehicles, classifies basic
behaviour, records events and sends notifications to mobile devices. All components are open source
and run locally in Docker.

## Features

- **Detection and tracking** of people and vehicles, each labelled with a stable identifier and confidence.
- **Number plate recognition** from full-resolution frames, checked against an authorised-vehicle list.
  UK plates are decoded for region of first registration and issue period.
- **Behaviour classification** from body pose and motion: walking, running, crouching, lying down,
  climbing, arms raised, loitering and entry into user-defined zones. Camera pan is compensated.
- **Recording** of annotated and original video while activity is present, with configurable retention.
- **Notifications** via ntfy, with annotated images, plate close-ups and video clips.
- **Web dashboard** with live view, activity log, vehicle register, recordings, zone editor and alert settings.
- **Health monitoring** with notifications for camera loss and recovery, processing delays, low disk space
  (oldest recordings are removed automatically) and system start-up.

## Architecture

| Service | Role |
|---|---|
| `frigate` | Camera ingest, RTSP restream, continuous recording, PTZ control |
| `vision` | Detection, tracking, pose, plate recognition, recording, notifications, dashboard |
| `ntfy` | Optional self-hosted notification server |
| `mqtt` | Message broker used by Frigate |

## Quick start

Requirements: Docker, and a PoE+ connection to the camera with RTSP and ONVIF enabled.

```bash
cp .env.example .env            # set camera address and credentials
./scripts/check_camera.sh       # verify connectivity and streams
docker compose up -d --build
```

- Dashboard: `http://<host>:8090` – available to any device on the local network; no internet connection is required
- Frigate: `https://<host>:8971` (initial password: `docker logs frigate`)
- Authorised vehicles: `settings/whitelist.yaml`
- Service settings: `docker-compose.yml` (`vision` environment)

To receive notifications, install the ntfy app and follow the steps in the dashboard's **Phone alerts** tab.

## Data and compliance

Recordings, logs and settings are stored locally in `recordings/`, `data/` and `settings/`. When the public
ntfy.sh server is used, notification content passes through that service; set `NTFY_SERVER=http://ntfy`
to keep all data on site. Operating CCTV and ANPR in the UK requires appropriate signage, a data
protection impact assessment, ICO registration and a defined retention policy.

## Open-source components

| Component | Purpose | Licence |
|---|---|---|
| [Frigate](https://github.com/blakeblackshear/frigate) | NVR, restreaming, recording | MIT |
| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Person and vehicle detection | Apache-2.0 |
| [Norfair](https://github.com/tryolabs/norfair) | Multi-object tracking | BSD-3-Clause |
| [RTMPose / MMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) | Human pose estimation | Apache-2.0 |
| [FastALPR](https://github.com/ankandrew/fast-alpr) | Number plate recognition pipeline | MIT |
| [open-image-models](https://github.com/ankandrew/open-image-models) | Plate detection | MIT |
| [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) | Plate character recognition | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Model inference | MIT |
| [OpenCV](https://github.com/opencv/opencv) | Video decoding and image processing | Apache-2.0 |
| [NumPy](https://github.com/numpy/numpy) | Numerical computing | BSD-3-Clause |
| [FFmpeg](https://ffmpeg.org/) | Recording and encoding | LGPL / GPL |
| [ntfy](https://github.com/binwiederhier/ntfy) | Push notifications | Apache-2.0 / GPL-2.0 |
| [Eclipse Mosquitto](https://github.com/eclipse-mosquitto/mosquitto) | MQTT broker | EPL-2.0 / EDL-1.0 |
| [Requests](https://github.com/psf/requests) | HTTP client | Apache-2.0 |
| [PyYAML](https://github.com/yaml/pyyaml) | Configuration files | MIT |
| [python-qrcode](https://github.com/lincolnloop/python-qrcode) | Subscription QR code | BSD-3-Clause |
| [SQLite](https://www.sqlite.org/) | Event storage | Public domain |
