import numpy as np
import cv2
from mtcnn_ort import MTCNN
import time
from flask import Flask, Response
import onnxruntime as rt
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

options = rt.SessionOptions()
options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

try:
    session = rt.InferenceSession('model.onnx', options, providers=EP_list)
    logging.info("Model ONNX chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle ONNX : {e}")
    raise ValueError(f"Erreur lors du chargement du modèle : {e}")

detector = MTCNN()

EMOJI = {
    0: cv2.imread('emojis/5-sad.png'),
    1: cv2.imread('emojis/6-surprise.png'),
    2: cv2.imread('emojis/3-happy.png'),
    3: cv2.imread('emojis/0-angry.png')
}

previous_emotion = None  # Dernière émotion prédite

frame_count = 0


def convert_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_face(image):
    start_time = time.time()
    faces = detector.detect_faces(image)
    elapsed_time = time.time() - start_time
    logging.info(f"Temps de détection (MTCNN) : {elapsed_time:.3f}s")
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


def process_face_every_n_frames(frame, n=3):
    global frame_count
    frame_count += 1
    if frame_count % n != 0:
        return None, None
    rgb_image = convert_to_rgb(frame)
    box = detect_face(rgb_image)

    if box is not None:
        cropped = crop_face(rgb_image, box)
        if cropped is None or cropped.size == 0:
            return None, None
        gray = convert_to_gray(cropped)
        processed = resize_and_normalize(gray)
        return processed, box
    return None, None


def run_inference(face):
    start_time = time.time()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: face})[0]
    elapsed_time = time.time() - start_time
    emotion = np.argmax(output)
    logging.info(f"Emotion prédite : {emotion} (Temps d'inférence ONNX : {elapsed_time:.3f}s)")
    return emotion


def update_display(frame, emotion):
    rows, cols, channels = EMOJI[emotion].shape
    frame[0:rows, 0:cols] = EMOJI[emotion]
    return cv2.flip(frame, 1)


def generate_frames(test_mode=False, num_frames=100):
    global previous_emotion

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    frame_count = 0
    start_time = time.time()

    while True:
        ret_val, frame = cam.read()
        if not ret_val:
            break

        face, box = process_face_every_n_frames(frame)

        if box is not None:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if face is not None:
            previous_emotion = run_inference(face)  # Met à jour l’émotion uniquement si le visage est détecté

        if previous_emotion is not None:
            frame = update_display(frame, previous_emotion)
        else:
            frame = cv2.flip(frame, 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if test_mode and frame_count >= num_frames:
            break

    cam.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(True), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
