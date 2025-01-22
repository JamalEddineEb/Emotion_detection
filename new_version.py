import numpy as np
import cv2
from mtcnn_ort import MTCNN
import time
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import onnxruntime as rt
import logging
import threading
import psutil  # Pour surveiller l'utilisation des ressources système

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

options = rt.SessionOptions()
options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

try:
    session = rt.InferenceSession('model.onnx', options, providers=EP_list)
    logging.info("Modèle ONNX chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle ONNX : {e}")
    raise ValueError(f"Erreur lors du chargement du modèle : {e}")

detector = MTCNN()

frame = None
lock = threading.Lock()

EMOJI = {
    0: cv2.imread('emojis/5-sad.png'),
    1: cv2.imread('emojis/6-surprise.png'),
    2: cv2.imread('emojis/3-happy.png'),
    3: cv2.imread('emojis/0-angry.png')
}

detection_time = 0
prediction_time = 0


def detect_face(image):
    global detection_time
    start_time = time.time()
    faces = detector.detect_faces(image)
    detection_time = time.time() - start_time
    if faces and faces[0]['confidence'] > 0.6:
        return faces[0]['box']
    return None


def crop_face(image, box):
    x, y, w, h = box
    return image[max(0, y-10):y+h+10, max(0, x-10):x+w+10]


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_and_normalize(image):
    resized = cv2.resize(image, (48, 48))
    normalized = resized.astype('float32') / 255
    return normalized.reshape(1, 48, 48, 1)


def run_inference(face):
    global prediction_time
    start_time = time.time()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: face})[0]
    prediction_time = time.time() - start_time
    emotion = int(np.argmax(output))
    return emotion


def video_processing():
    global frame
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        ret, img = cam.read()
        if not ret:
            break

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        box = detect_face(rgb_image)

        if box:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cropped = crop_face(rgb_image, box)
            gray = convert_to_gray(cropped)
            normalized_face = resize_and_normalize(gray)

            # Effectue l'inférence d'émotion
            emotion = run_inference(normalized_face)
            rows, cols, _ = EMOJI[emotion].shape
            img[0:rows, 0:cols] = EMOJI[emotion]

        with lock:
            frame = img

    cam.release()


def system_monitoring():
    while True:
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent

        # Envoi des données toutes les 400 millisecondes
        socketio.emit('update_system', {
            'cpu': cpu_usage,
            'memory': memory_usage,
            'detection_time': detection_time,
            'prediction_time': prediction_time
        })

        time.sleep(0.4)  # Attente de 400 millisecondes


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global frame
        while True:
            with lock:
                if frame is None:
                    continue
                _, buffer = cv2.imencode('.jpg', frame)
                img_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    thread_video = threading.Thread(target=video_processing, daemon=True)
    thread_video.start()

    thread_monitoring = threading.Thread(target=system_monitoring, daemon=True)
    thread_monitoring.start()

    socketio.run(app, host='0.0.0.0', port=5000)