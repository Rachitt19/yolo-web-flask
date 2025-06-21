from flask import Flask, render_template, Response, request, send_file, jsonify
from ultralytics import YOLO
import cv2
from PIL import Image
import io
import numpy as np

app = Flask(__name__)
model = YOLO('yolov8n.pt')
camera = cv2.VideoCapture(0)

last_detected_classes = []

def confidence_color(conf):
    if conf > 0.7:
        return (0, 255, 0)  # Green
    elif conf > 0.4:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 0, 255)  # Red

def gen_frames():
    global last_detected_classes
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model(frame)[0]
            classes_in_frame = []
            top_label = "None"
            top_conf = 0

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                color = confidence_color(conf)
                classes_in_frame.append(model.names[cls])
                if conf > top_conf:
                    top_label = model.names[cls]
                    top_conf = conf

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            last_detected_classes = list(set(classes_in_frame))  # unique classes detected this frame

            if top_conf > 0:
                tag = f"Detected: {top_label} ({top_conf * 100:.1f}%)"
            else:
                tag = "No detection"

            # draw tag on image
            cv2.putText(frame, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    image = Image.open(file.stream).convert("RGB")
    image_np = np.array(image)

    results = model(image_np)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"
        color = confidence_color(conf)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_np, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    result_image = Image.fromarray(image_np)
    img_io = io.BytesIO()
    result_image.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

# New endpoint to serve detected classes for voice feedback & alerts
@app.route('/detected_classes')
def detected_classes():
    global last_detected_classes
    return jsonify(last_detected_classes)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)