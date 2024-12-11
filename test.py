import onnxruntime as rt
import cv2
import numpy as np

class MTCNN_ONNX:
    def __init__(self, pnet_path, rnet_path, onet_path):
        # Load the ONNX models for P-Net, R-Net, O-Net
        self.pnet_session = rt.InferenceSession(pnet_path)
        self.rnet_session = rt.InferenceSession(rnet_path)
        self.onet_session = rt.InferenceSession(onet_path)

    def detect_faces(self, image):
        # Convert image to RGB (for ONNX model compatibility)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces using P-Net
        pnet_boxes = self._pnet_detect(image_rgb)
        faces = []

        for box in pnet_boxes:
            # Crop and detect with R-Net
            cropped_face = self._crop_face(image_rgb, box)
            rnet_boxes = self._rnet_detect(cropped_face)
            for rbox in rnet_boxes:
                # Crop and detect with O-Net
                cropped_face = self._crop_face(image_rgb, rbox)
                onet_boxes = self._onet_detect(cropped_face)
                for obox in onet_boxes:
                    faces.append(obox)
        
        return faces

    def _pnet_detect(self, image):
        input_name = self.pnet_session.get_inputs()[0].name
        output_name = self.pnet_session.get_outputs()[0].name
        output = self.pnet_session.run([output_name], {input_name: image})[0]
        # Here you would process the output from P-Net
        return self._parse_boxes(output)

    def _rnet_detect(self, face):
        input_name = self.rnet_session.get_inputs()[0].name
        output_name = self.rnet_session.get_outputs()[0].name
        output = self.rnet_session.run([output_name], {input_name: face})[0]
        return self._parse_boxes(output)

    def _onet_detect(self, face):
        input_name = self.onet_session.get_inputs()[0].name
        output_name = self.onet_session.get_outputs()[0].name
        output = self.onet_session.run([output_name], {input_name: face})[0]
        return self._parse_boxes(output)

    def _parse_boxes(self, output):
        # Parse the output boxes
        # You can modify this to handle your model's output format
        boxes = []
        for i in range(output.shape[0]):
            confidence = output[i, 0]
            if confidence > 0.5:
                # Box format [x1, y1, x2, y2]
                boxes.append(output[i, 1:5])
        return boxes

    def _crop_face(self, image, box):
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]

# Usage
if __name__ == "__main__":
    # Paths to your ONNX model files
    pnet_path = "pnet.onnx"
    rnet_path = "rnet.onnx"
    onet_path = "onet.onnx"

    mtcnn = MTCNN_ONNX(pnet_path, rnet_path, onet_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = mtcnn.detect_faces(frame)

        # Draw boxes on the detected faces
        for box in faces:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show the frame with detected faces
        cv2.imshow('MTCNN with ONNX', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
