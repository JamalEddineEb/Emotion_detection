# Emotion_detection

A small Python project for real-time facial emotion detection using MTCNN for face detection and an ONNX emotion classifier. The app captures webcam frames, detects faces, runs emotion inference, and overlays emoji icons on the video stream. A lightweight Flask endpoint serves the camera stream as MJPEG.

**Key features**
- **Real-time webcam inference**: captures frames from a webcam and processes them live.
- **Face detection**: MTCNN implementation in `mtcnn_ort/` using ONNX models (`pnet.onnx`, `rnet.onnx`, `onet.onnx`).
- **ONNX emotion model**: multiple model formats are present (ONNX, TFLite, Keras) — main ONNX model used by the examples.
- **Performance monitoring**: `performance_monitor.py` collects timing and memory consumption per pipeline stage.
- **Flask preview**: the app serves video as MJPEG so you can view the annotated stream in a browser.

## Repository layout (important files)
- `real_time.py` — main root example: loads `model.onnx`, serves MJPEG on `/` (port 5000).
- `src/real_time.py` — alternative entry referencing `models/model.onnx` and route `/video_feed`.
- `mtcnn_ort/` — MTCNN face detector implementation and ONNX runners (`pnet.onnx`, `rnet.onnx`, `onet.onnx`).
- `models/` and root — contains saved models: `model.onnx`, `model.tflite`, `model.h5`, and `ModelX/` (TensorFlow SavedModel files).
- `emojis/` — emoji images used to overlay predictions on the video frames.
- `performance_monitor.py` — simple perf monitor capturing time and memory per pipeline stage.
- `requirements.txt` — Python dependencies used by the project.

## Requirements
- Python 3.8+ recommended.
- The project lists dependencies in `requirements.txt`. Install them in a virtual environment:

```fish
python -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
```

Notes:
- If you want GPU-accelerated ONNX inference, install `onnxruntime-gpu` instead of `onnxruntime` and ensure you have compatible CUDA/cuDNN drivers.
- The project uses `flask`, `opencv-python` (via `cv2` import), `onnxruntime`, and `psutil` (used by the performance monitor). If `opencv-python` is missing, install it with `pip install opencv-python`.

## Running the demo (web preview)
1. Make sure a webcam is connected and accessible (device index 0). If your camera is different, edit the `cv2.VideoCapture(0)` call.
2. Start the app (from repository root):

```fish
python real_time.py
```

3. Open a browser and go to `http://0.0.0.0:5000/` (root `real_time.py`).

What happens:
- The Flask app returns an MJPEG stream.
- Each camera frame is passed through MTCNN; detected faces are cropped, converted to grayscale, resized to 48x48, normalized, and run through the ONNX model.
- The predicted emotion index is used to overlay an emoji at the top-left of the frame.

## Switching models / paths
- Root `real_time.py` expects `model.onnx` in the repository root. `src/real_time.py` expects `models/model.onnx`.
- You can replace the ONNX model with any compatible model that accepts the same input shape and produces a compatible output probability vector.

## Performance & Debugging
- `performance_monitor.py` collects per-stage timing and memory. The demo prints metrics when run in test mode, or you can call `monitor.print_metrics()` manually.
- If the ONNX model fails to load, check the provider configuration. By default the code attempts `['CUDAExecutionProvider','CPUExecutionProvider']`. If you don't have GPU-supporting onnxruntime installed, install CPU-only `onnxruntime` or switch providers to `['CPUExecutionProvider']` in the script.
- If camera capture fails, ensure no other app is using the webcam and the device index is correct.
- If emoji images fail to load, ensure the `emojis/` folder is populated and filenames match those referenced in the script.

## Notes for development
- The `mtcnn_ort` package is included in the repo for convenience. It uses ONNX-based PNet/RNet/ONet models for detection.
- There are several model formats present (Keras H5, TFLite, ONNX, and SavedModel). The provided inference code uses ONNXRuntime for portability.

