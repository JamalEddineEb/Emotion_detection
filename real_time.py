import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from mtcnn_cv2 import MTCNN

# Initialize MTCNN for face detection
detector = MTCNN()

# Load and initialize TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully.")

EMOJI = {
    0: cv2.imread('emojis/5-sad.png'),
    1: cv2.imread('emojis/6-surprise.png'),
    2: cv2.imread('emojis/3-happy.png'),
    3: cv2.imread('emojis/0-angry.png')
}

def localize_face(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x, y, w, h = detector.detect_faces(image)[0]['box']
    while detector.detect_faces(image)[0]['confidence']>0.6:
        cropped = image[y-10:y+h+10, x-10:x+w+10]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray,(48,48)).astype('float32') / 255  # Changed to float32
        localized = resized.reshape(1,48,48,1)
        return localized, [x, y, w, h]

def detection_emotion():
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, frame = cam.read()

        # Localisation des visages dans l'image
        face, box = localize_face(frame)
        
        # Reconnaissance de l'émotion with TFLite
        interpreter.set_tensor(input_details[0]['index'], face)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        emotion = np.argmax(output)

        # Création d'une ROI pour l'affichage du smiley
        rows,cols,channels = EMOJI[0].shape
        roi_smiley = frame[0:rows, 0:cols]

        # Selon le résultat de la prédiction
        frame[0:rows, 0:cols ] = EMOJI[emotion]
 
        # Retourne l'image pour l'afficer en miroir
        frame = cv2.flip(frame, 1)
        
        # Affichage de l'image
        cv2.imshow('How are you ?', frame)
        
        # Pour quitter, appuyez sur la touche ESC
        if cv2.waitKey(1) == 27: 
            break 
    
    cam.release()
    cv2.destroyAllWindows()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(48,48), color_mode='grayscale')
    img_array = img_to_array(img).astype('float32') / 255.0  # Changed to float32
    print(np.array(img_array).shape)
    return np.expand_dims(img_array, axis=0)

# For testing single image
def test_single_image(image_path):
    image = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    emotion = np.argmax(output)
    print(emotion)
    return emotion

# Run real-time detection
if __name__ == "__main__":
    detection_emotion()