# real_time.py
import tensorflow as tf
import numpy as np
import cv2
import time
from performance_monitor import PerformanceMonitor
from detectors import MTCNNFaceDetector, YOLOFaceDetector

monitor = PerformanceMonitor()

# Initialize components
detector = MTCNNFaceDetector()
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

EMOJI = {
    0: cv2.imread('assets/emojis/5-sad.png'),
    1: cv2.imread('assets/emojis/6-surprise.png'),
    2: cv2.imread('assets/emojis/3-happy.png'),
    3: cv2.imread('assets/emojis/0-angry.png')
}

@monitor.measure('color_conversion')
def convert_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

@monitor.measure('face_detection') 
def detect_face(image):
    return detector.detect_face(image)
@monitor.measure('face_cropping')
def crop_face(image, box):
    x, y, w, h = box
    return image[y-10:y+h+10, x-10:x+w+10]

@monitor.measure('grayscale_conversion')
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

@monitor.measure('resize_normalize')
def resize_and_normalize(image):
    resized = cv2.resize(image, (48, 48))
    normalized = resized.astype('float32') / 255
    return normalized.reshape(1, 48, 48, 1)

def process_face(frame):
    rgb_image = convert_to_rgb(frame)
    box = detect_face(rgb_image)
    
    if box is not None:
        cropped = crop_face(rgb_image, box)
        gray = convert_to_gray(cropped)
        processed = resize_and_normalize(gray)
        return processed, box
    return None, None

@monitor.measure('inference')
def run_inference(face):
    interpreter.set_tensor(input_details[0]['index'], face)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output)

@monitor.measure('display')
def update_display(frame, emotion):
    rows, cols, _ = EMOJI[0].shape
    frame[0:rows, 0:cols] = EMOJI[emotion]
    return cv2.flip(frame, 1)

def detection_emotion(test_mode=False, num_frames=100):
    cam = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret_val, frame = cam.read()
        if not ret_val:
            break
            
        frame_count += 1
        
        face, box = process_face(frame)
        
        if face is not None:
            emotion = run_inference(face)
            frame = update_display(frame, emotion)
            
        cv2.imshow('How are you ?', frame)
        
        if cv2.waitKey(1) == 27 or (test_mode and frame_count >= num_frames):
            break
    
    duration = time.time() - start_time
    
    if test_mode:
        monitor.print_metrics(duration, frame_count)
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detection_emotion(test_mode=True)