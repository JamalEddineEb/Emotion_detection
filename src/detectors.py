import cv2
from mtcnn_cv2 import MTCNN

class FaceDetectorBase:
    def detect_face(self, image):
        raise NotImplementedError

class MTCNNFaceDetector(FaceDetectorBase):
    def __init__(self):
        self.detector = MTCNN()
        
    def detect_face(self, image):
        faces = self.detector.detect_faces(image)
        if faces and faces[0]['confidence'] > 0.6:
            return faces[0]['box']
        return None

class YOLOFaceDetector(FaceDetectorBase):
    def __init__(self, weights_path, cfg_path):
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        
    def detect_face(self, image):
        # YOLO implementation here
        pass