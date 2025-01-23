import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import lgpio

# LED setup
LED_pin = 16  # GPIO pin 16
h = lgpio.gpiochip_open(0)  # Open the GPIO chip
lgpio.gpio_claim_output(h, LED_pin)  # Set the pin as output

def control_led():
    """Function to turn the LED on for 3 seconds and then off"""
    lgpio.gpio_write(h, LED_pin, 1)  # Turn the LED on
    time.sleep(5)  # Wait for 3 seconds
    lgpio.gpio_write(h, LED_pin, 0)  # Turn the LED off

# LEDController class to handle LED activation in a separate thread
class LEDController(Thread):
    def __init__(self):
        super().__init__()
        self.led_on = False
        self.daemon = True

    def run(self):
        # Continuously check for LED activation trigger
        while True:
            if self.led_on:
                control_led()
                self.led_on = False

    def trigger(self):
        # Trigger the LED activation
        self.led_on = True

# VideoStream class to handle the webcam stream
class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Start the video stream in a separate thread
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Continuously read frames from the webcam
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the latest frame
        return self.frame

    def stop(self):
        # Stop the video stream
        self.stopped = True

# Command-line arguments parsing
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', required=True, help='Folder the .tflite file is located in')
parser.add_argument('--graph', default='detect.tflite', help='Name of the .tflite file')
parser.add_argument('--labels', default='labelmap.txt', help='Name of the labelmap file')
parser.add_argument('--threshold', default=0.5, help='Minimum confidence threshold for displaying detected objects')
parser.add_argument('--resolution', default='1280x720', help='Webcam resolution in WxH')
parser.add_argument('--edgetpu', action='store_true', help='Use Coral Edge TPU')
args = parser.parse_args()

# Load model and labelmap configurations
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# TensorFlow Lite interpreter setup
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Adjust model file for Edge TPU if required
if use_TPU:
    if GRAPH_NAME == 'detect.tflite':
        GRAPH_NAME = 'edgetpu.tflite'

# Paths for the model and labels
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load labels
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del labels[0]

# Initialize the interpreter
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Determine output tensor indexes
outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize FPS calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Start the video stream and LED controller
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

led_controller = LEDController()
led_controller.start()

try:
    while True:
        # Measure start time for FPS calculation
        t1 = cv2.getTickCount()

        # Read the current frame and preprocess
        frame1 = videostream.read()
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize input data for floating-point models
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform inference with the model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Extract detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        # Identify the object with the highest score
        highest_score_index = np.argmax(scores)
        if scores[highest_score_index] > 0.95:
            # Draw bounding box and label for detected object
            ymin = int(max(1, (boxes[highest_score_index][0] * imH)))
            xmin = int(max(1, (boxes[highest_score_index][1] * imW)))
            ymax = int(min(imH, (boxes[highest_score_index][2] * imH)))
            xmax = int(min(imW, (boxes[highest_score_index][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[highest_score_index])]
            label = '%s: %d%%' % (object_name, int(scores[highest_score_index] * 100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Trigger LED control
            led_controller.trigger()

        # Display FPS and frame in fullscreen mode
        cv2.namedWindow('Object detector', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Object detector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Object detector', frame)

        # Measure end time for FPS calculation
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by User")

finally:
    # Release resources and close GPIO
    cv2.destroyAllWindows()
    videostream.stop()
    lgpio.gpiochip_close(h)
  
