FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY plate_watcher ./plate_watcher
# Download the plate models at build time so the box works offline afterwards.
RUN python -c "from plate_watcher.reader import PlateReader; PlateReader()"
CMD ["python", "-u", "-m", "plate_watcher"]
