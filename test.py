import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

import os
import sys
import numpy as np
import cv2
from pprint import pprint

import matplotlib.pyplot as plt
from mtcnn_cv2 import MTCNN

detector = MTCNN()



recognator = tf.saved_model.load("./ModelX")
print("Model loaded successfully.")

EMOJI = {
 0: cv2.imread('emojis/5-sad.png'),
 1: cv2.imread('emojis/6-surprise.png'),
 2: cv2.imread('emojis/3-happy.png'),
 3: cv2.imread('emojis/0-angry.png')
 }

TEST_IMAGE_PATH = "./images/test.png"

# Fonction de localisation des visages - modifications pour utilisation avec flux video filmant 1 seule personne
def localize_face(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x, y, w, h = detector.detect_faces(image)[0]['box']
    while detector.detect_faces(image)[0]['confidence']>0.6 :
        cropped = image[y-10:y+h+10, x-10:x+w+10]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray,(48,48)).astype('float') / 255
        localized = resized.reshape(1,48,48,1)
        return localized, [x, y, w, h]


def detection_emotion():

    cam = cv2.VideoCapture(0)
    while True:
        ret_val, frame = cam.read()

        # Localisation des visages dans l'image
        face, box = localize_face(frame)
        
        # Reconnaissance de l'émotion
        emotion = np.argmax(recognator(face))

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

with open(TEST_IMAGE_PATH,'r') as f:
    face = f

def preprocess_image(image_path):
    # Load the image and resize it
    img = load_img(image_path, target_size=(48,48), color_mode='grayscale')
    # img = load_img(image_path, target_size=target_size, color_mode='grayscale')
    img_array = img_to_array(img) / 255.0  # Normalize the image to [0, 1]
    print(np.array(img_array).shape)
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

image = preprocess_image(TEST_IMAGE_PATH)



emotion = np.argmax(recognator(image))
print(emotion)