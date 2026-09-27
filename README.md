# Real-Time Object Detection & Tracking on Raspberry Pi

Real-time object detection and lightweight object tracking using YOLO11n, OpenCV, Python, and Raspberry Pi 3B+.

## Overview

The system streams a laptop webcam over Wi-Fi to a Raspberry Pi 3B+. The Raspberry Pi performs YOLO object detection and lightweight centroid-based tracking.

The pipeline uses asynchronous frame capture so camera acquisition continues while the CPU performs neural-network inference.

## Architecture

Laptop Webcam
    |
    | MJPEG over Wi-Fi
    v
Raspberry Pi 3B+
    |
    +-- Async frame capture
    |
    +-- Latest-frame buffer
    |
    +-- YOLO11n detection
    |       |
    |       +-- 320x320 input
    |       +-- confidence filtering
    |
    +-- Centroid tracking
    |
    +-- Position prediction
    |
    +-- Target management

## Features

- YOLO11n object detection
- OpenCV video processing
- Asynchronous camera capture
- Latest-frame buffering
- Periodic YOLO inference
- Class-aware centroid tracking
- Persistent object IDs
- Lightweight position prediction
- Automatic target selection
- Target loss detection
- Target reacquisition
- Headless Raspberry Pi operation
- Terminal performance statistics

## Hardware

- Raspberry Pi 3B+
- 1 GB RAM
- Laptop webcam
- Wi-Fi network

Current setup:

Laptop webcam -> MJPEG stream -> Wi-Fi -> Raspberry Pi

## Software

- Python 3
- OpenCV
- Ultralytics YOLO
- YOLO11n
- PyTorch
- NumPy
- Linux
- Git

## Current Configuration

Main tracker:

src/detect_camera.py

- YOLO model: YOLO11n
- YOLO input: 320x320
- Detection interval: every 8 captured frames
- Confidence threshold: 0.50
- Maximum matching distance: 120 pixels
- Maximum missed detections: 5
- Maximum prediction step: 25 pixels
- Camera frame size: 640x480

## Performance

Measured on Raspberry Pi 3B+:

| Metric | Result |
|---|---:|
| Camera capture | ~15.0 FPS |
| Pipeline loop | ~5.27 FPS |
| YOLO detection rate | ~0.64 FPS |
| YOLO-only throughput | ~0.79 FPS |
| Average YOLO inference | ~1265 ms |
| YOLO input | 320x320 |
| Frames processed | 264 |
| YOLO detections | 32 |

The camera capture thread continues independently at approximately 15 FPS while YOLO inference runs on the latest available frame.

## Tracking

Objects are represented by bounding-box centroids.

The tracker matches detections using:

- object class
- centroid distance
- maximum matching distance

Between YOLO detections, recent motion is used to estimate the object's position.

The system also handles target loss and reacquisition.

## Running

### Laptop

Start the webcam stream:

python src/stream_camera.py

### Raspberry Pi

SSH into the Pi:

ssh varun@192.168.1.8

Then:

cd ~/yolo-rpi-tracker
source .venv/bin/activate
python src/detect_camera.py

The tracker runs headlessly and prints detection and tracking statistics in the terminal.

## Benchmarking

Camera benchmark:

python src/camera_benchmark.py

Performance benchmark:

python src/performance_benchmark.py

These benchmarks should be run on the Raspberry Pi when measuring Pi performance.

## Project Structure

yolo-rpi-tracker/
├── .gitignore
├── README.md
├── src/
│   ├── camera_benchmark.py
│   ├── detect_camera.py
│   ├── performance_benchmark.py
│   └── stream_camera.py
└── .venv/

.venv/ and YOLO model weights are excluded from Git.

## Why Asynchronous Capture?

Synchronous camera capture caused the camera pipeline to block while YOLO inference was running.

The asynchronous design separates:

camera capture -> latest frame -> YOLO -> tracker

This prevents old frames from accumulating and keeps the input stream responsive.

## Why 320x320?

On the Raspberry Pi 3B+, YOLO inference at 640x640 was substantially slower than at 320x320.

Measured benchmark:

- 640x640: ~2669 ms/inference
- 320x320: ~807 ms/inference

The lower resolution provides substantially better throughput on the constrained CPU.

## Limitations

- Raspberry Pi 3B+ CPU inference is slow.
- YOLO detection runs much slower than camera capture.
- Small objects may be missed at 320x320.
- The current camera is network-streamed from a laptop.
- Detection accuracy depends on the pretrained YOLO model.

## Future Improvements

- Model quantization
- ONNX or other optimized inference backends
- Hardware acceleration
- Improved motion prediction
- Kalman filtering
- Direct Raspberry Pi camera integration
- C++ implementation of performance-critical components
- Robotics actuator integration
