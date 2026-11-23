# syntax=docker/dockerfile:1.7
FROM python:3.12-slim
WORKDIR /app
# ffmpeg: saves recordings by copying the camera video (no re-encoding)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# YOLOX-S people + vehicle detector (Apache-2.0), from the official YOLOX release
ADD --checksum=sha256:c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063 \
    https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx /models/yolox_s.onnx
# RTMPose-m body pose model (Apache-2.0, OpenMMLab) – for guessing what people are doing
ADD --checksum=sha256:f7fbb6c5c11a1bb70f3d445e4ddec5d144ea89ad5649c081ef976f9b24a0b741 \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip /tmp/rtmpose.zip
RUN python -c "import zipfile; z = zipfile.ZipFile('/tmp/rtmpose.zip'); \
n = [f for f in z.namelist() if f.endswith('end2end.onnx')][0]; open('/models/rtmpose-m.onnx', 'wb').write(z.read(n))" \
    && rm /tmp/rtmpose.zip
COPY vision ./vision
# Download the plate models at build time so it works offline afterwards
RUN python -c "from vision.plates import PlateReader; PlateReader()"
# Camera video: TCP (UDP drops packets → smeared frames) + no buffering (lower delay)
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
EXPOSE 8090
CMD ["python", "-u", "-m", "vision"]
