import numpy as np
import tvm
from tvm import relay
from tvm.driver import tvmc
from tvm.contrib import graph_executor
import torchvision.transforms as transforms
from PIL import Image
import logging
import onnx
import time
import cv2
from mtcnn_ort import MTCNN
import time
from flask import Flask, Response
from performance_monitor import PerformanceMonitor
import onnxruntime as rt
from tvm import autotvm
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner,GATuner,RandomTuner,GridSearchTuner

print('1. Get ONNX model')
onnx_model = onnx.load('models/model.onnx')

inputs = onnx_model.graph.input

for input in inputs:
    print(f"Input name: {input.name}")
    print(f"Input shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")

print('2. Convert to TVM Relay')
shape_dict = {"module_wrapper_input": [1, 48, 48, 1]}
mod, params = relay.frontend.from_onnx(onnx_model,shape_dict)
print('3. Compile for target')
target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
# target = tvm.target.arm_cpu()

# tune_tasks = autotvm.task.extract_from_program(mod, target=target,params=params)

# # Use autotvm to tune the tasks
# tuner = autotvm.tuner.XGBTuner(tune_tasks)
# tuner.tune(n_trial=100)  # Number of tuning trials to run

target_host = "llvm"
tgt = tvm.target.Target(target, host=target_host)

with tvm.transform.PassContext(opt_level=1):
    lib = relay.build(mod, target=tgt, params=params)


print('4. Export the model')
lib.export_library("compiled-onnx-llvm-model.so")
#lib = tvm.runtime.load_module("compiled-onnx-llvm-model.so")


print('5. Create an executor')
device = tvm.device(target, 0)
module = graph_executor.GraphModule(lib["default"](device))

number = 10
repeat = 1
min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
timeout = 10  # in seconds

# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

tuning_option = {
    "tuner": "xgb",
    "trials": 20,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "resnet-50-v2-autotuning.json",
}

tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

    # choose tuner
    tuner = "xgb"

    # create tuner
    if tuner == "xgb":
        tuner_obj = XGBTuner(task, loss_type="reg")
    elif tuner == "xgb_knob":
        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="knob")
    elif tuner == "xgb_itervar":
        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="itervar")
    elif tuner == "xgb_curve":
        tuner_obj = XGBTuner(task, loss_type="reg", feature_type="curve")
    elif tuner == "xgb_rank":
        tuner_obj = XGBTuner(task, loss_type="rank")
    elif tuner == "xgb_rank_knob":
        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="knob")
    elif tuner == "xgb_rank_itervar":
        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="itervar")
    elif tuner == "xgb_rank_curve":
        tuner_obj = XGBTuner(task, loss_type="rank", feature_type="curve")
    elif tuner == "xgb_rank_binary":
        tuner_obj = XGBTuner(task, loss_type="rank-binary")
    elif tuner == "xgb_rank_binary_knob":
        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="knob")
    elif tuner == "xgb_rank_binary_itervar":
        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="itervar")
    elif tuner == "xgb_rank_binary_curve":
        tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="curve")
    elif tuner == "ga":
        tuner_obj = GATuner(task, pop_size=50)
    elif tuner == "random":
        tuner_obj = RandomTuner(task)
    elif tuner == "gridsearch":
        tuner_obj = GridSearchTuner(task)
    else:
        raise ValueError("Invalid tuner: " + tuner)

    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )

with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

app = Flask(__name__)

# # Specify GPU execution provider
# EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # Fallback to CPU if GPU is unavailable

# # Load the ONNX model with GPU support
# try:
#     session = rt.InferenceSession('model.onnx', options, providers=EP_list)
#     print("Model loaded successfully with GPU execution provider.")
# except Exception as e:
#     raise ValueError(f"Failed to load the model: {e}")

# # Check which providers are being used
# print("Available providers:", rt.get_available_providers())
# print("Using providers:", session.get_providers())


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
    module.set_input(input.name, face)
    module.run()
    output = module.get_output(0).asnumpy() 
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
    return Response(generate_frames(True), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
