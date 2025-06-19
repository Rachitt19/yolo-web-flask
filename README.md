# YOLOv8 Real-Time Object Detection Web App

A Flask-based web application that performs real-time multi-class object detection using the YOLOv8 model. The app supports webcam streaming and image upload modes, provides voice feedback for detected objects, voice command controls, and visual/audio alerts on person detection.

---

## Features

- Real-time object detection on webcam video feed using YOLOv8.
- Image upload mode for detecting objects in static images.
- Voice feedback announcing detected object classes.
- Voice commands to pause/resume detection and switch modes.
- Visual alert box and alarm sound on person detection.
- Confidence-based colored bounding boxes.
- Responsive, clean UI with webcam and upload controls.

---

## Demo

*(Add link to your deployed app here once available)*

---

## Requirements

- Python 3.8+
- Flask
- OpenCV (`opencv-python`)
- Ultralytics YOLO (`ultralytics`)
- Pillow
- NumPy

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/yolov8-flask-detection.git
cd yolov8-flask-detection
