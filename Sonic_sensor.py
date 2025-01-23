import lgpio
import time
import os

# Ultrasonic sensor pin configuration
TRIG_PIN = 23  # Trig pin of the ultrasonic sensor
ECHO_PIN = 24  # Echo pin of the ultrasonic sensor
DISTANCE_THRESHOLD = 30  # Detection distance threshold (cm)

# Initialize GPIO pins
h = lgpio.gpiochip_open(0)  # Open the default GPIO chip
lgpio.gpio_claim_output(h, TRIG_PIN)  # Set TRIG_PIN as output
lgpio.gpio_claim_input(h, ECHO_PIN)   # Set ECHO_PIN as input

# Function to measure distance
def measure_distance():
    lgpio.gpio_write(h, TRIG_PIN, 1)
    time.sleep(0.00001)  # 10μs pulse
    lgpio.gpio_write(h, TRIG_PIN, 0)

    start_time = time.time()
    stop_time = time.time()

    while lgpio.gpio_read(h, ECHO_PIN) == 0:
        start_time = time.time()

    while lgpio.gpio_read(h, ECHO_PIN) == 1:
        stop_time = time.time()

    # Calculate elapsed time and convert to distance
    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2  # Speed of sound: 343m/s
    return distance

# Function to execute the TFLite command
def run_tflite_command():
    print("Object detected within threshold distance. Running TFLite...")
    os.system("python3 TFLite_detection_webcam.py --modeldir=custom_model_lite")

try:
    print("Initializing ultrasonic sensor...")
    time.sleep(2)

    while True:
        distance = measure_distance()
        print(f"Measured distance: {distance:.2f} cm")

        if distance < DISTANCE_THRESHOLD:
            run_tflite_command()
            print("TFLite execution complete. Waiting 5 seconds before resuming detection...")
            time.sleep(5)  # Wait 5 seconds after execution
        else:
            print("Distance is greater than the threshold. Standing by...")
        
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Exiting program...")
finally:
    lgpio.gpiochip_close(h)  # Clean up GPIO pins
