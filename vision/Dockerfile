# syntax=docker/dockerfile:1.7
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# YOLOX-S people + vehicle detector (Apache-2.0), from the official YOLOX release
ADD --checksum=sha256:c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063 \
    https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx /models/yolox_s.onnx
COPY vision ./vision
# Download the plate models at build time so it works offline afterwards
RUN python -c "from vision.plates import PlateReader; PlateReader()"
# Camera video: TCP (UDP drops packets → smeared frames) + no buffering (lower delay)
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
EXPOSE 8090
CMD ["python", "-u", "-m", "vision"]
