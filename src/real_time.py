import numpy as np
import cv2
from mtcnn_ort import MTCNN
import time
from flask import Flask, Response
from performance_monitor import PerformanceMonitor
import onnxruntime as rt

# Configure session options
options = rt.SessionOptions()
options.graph_optimization_level= rt.GraphOptimizationLevel.ORT_ENABLE_ALL
# options.enable_profiling = True
app = Flask(__name__)

# Specify GPU execution provider
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # Fallback to CPU if GPU is unavailable

# Load the ONNX model with GPU support
try:
    session = rt.InferenceSession('models/model.onnx', options, providers=EP_list)
    print("Model loaded successfully with GPU execution provider.")
except Exception as e:
    raise ValueError(f"Failed to load the model: {e}")

# Check which providers are being used
print("Available providers:", rt.get_available_providers())
print("Using providers:", session.get_providers())


monitor = PerformanceMonitor()

# Initialize components
detector = MTCNN()


EMOJI = {
    0: cv2.imread('emojis/5-sad.png'),
    1: cv2.imread('emojis/6-surprise.png'),
    2: cv2.imread('emojis/3-happy.png'),
    3: cv2.imread('emojis/0-angry.png')
}

# @monitor.measure('color_conversion')
def convert_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

@monitor.measure('mtcnn_detection')
def detect_face(image):
    faces = detector.detect_faces(image)
    if faces and faces[0]['confidence'] > 0.6:
        return faces[0]['box']
    return None

# @monitor.measure('face_cropping')
def crop_face(image, box):
    x, y, w, h = box
    return image[y-10:y+h+10, x-10:x+w+10]

# @monitor.measure('grayscale_conversion')
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# @monitor.measure('resize_normalize')
def resize_and_normalize(image):
    resized = cv2.resize(image, (48, 48))
    normalized = resized.astype('float32') / 255
    return normalized.reshape(1, 48, 48, 1)

def process_face(frame):
    rgb_image = convert_to_rgb(frame)
    box = detect_face(rgb_image)
    
    if box is not None:
        cropped = crop_face(rgb_image, box)
        if cropped is None or cropped.size == 0:
            return None,None
        gray = convert_to_gray(cropped)
        processed = resize_and_normalize(gray)
        return processed, box
    return None, None

@monitor.measure('inference')
def run_inference(face):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: face})[0]
    return np.argmax(output)


# @monitor.measure('display')
def update_display(frame, emotion):
    # Define the position where the emoji will be placed
    x, y = 10, 10  # Top-left corner of the frame

    # Get the corresponding emoji image
    rows,cols,channels = EMOJI[0].shape
    roi_smiley = frame[0:rows, 0:cols]

    # Selon le résultat de la prédiction
    frame[0:rows, 0:cols ] = EMOJI[emotion]

    # Flip the frame horizontally
    return cv2.flip(frame, 1)




def generate_frames(test_mode=False, num_frames=100):
    cam = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time()

    while True:
        ret_val, frame = cam.read()
        if not ret_val:
            break

        frame_count += 1
        face, box = process_face(frame)

        if box is not None:
            # Draw the bounding box
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if face is not None:
            emotion = run_inference(face)
            frame = update_display(frame, emotion)
        else:
            frame = cv2.flip(frame, 1)


        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as a multipart HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if test_mode and frame_count >= num_frames:
            monitor.print_metrics(time.time()-start_time,num_frames)
            break

    cam.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
