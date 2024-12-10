from ultralytics import YOLO
import cv2
import numpy as np
import RPi.GPIO as GPIO
import serial

# Setup GPIO for the LED
LED_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# Setup Serial for HuskyLens
SERIAL_PORT = "/dev/serial0"  # Default serial port for Raspberry Pi GPIO
BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # Replace "best.pt" with the path to your trained model file

# Function to get image from HuskyLens
def get_huskylens_image():
    # Read data from the HuskyLens (depends on how the image data is sent)
    # If the HuskyLens provides a frame, adapt this accordingly.
    # This example assumes an OpenCV-compatible image is sent.
    if ser.in_waiting > 0:
        data = ser.read(ser.in_waiting)
        # Process the data to convert to an image (depends on HuskyLens interface)
        # Placeholder for actual image retrieval logic
        # For now, assume we have a preloaded image as a test
        frame = cv2.imread("huskylens_sample.jpg")  # Replace with actual image retrieval
        return frame
    return None

# Main Loop
try:
    while True:
        # Get an image from the HuskyLens
        frame = get_huskylens_image()
        if frame is None:
            print("No image received from HuskyLens.")
            continue

        # Convert the frame to RGB (YOLOv8 expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLOv8 inference on the frame
        results = model.predict(frame_rgb, conf=0.3)  # Adjust `conf` (confidence threshold) if needed

        # Extract detections
        detections = results[0].boxes.data.cpu().numpy()  # Array of [class, confidence, x1, y1, x2, y2]

        # Check for weeds (class 0 assumed as weed) and confidence > 0.2
        weed_detected = any(detection[1] > 0.2 and detection[0] == 0 for detection in detections)

        # Turn LED ON if weed detected, else turn it OFF
        if weed_detected:
            GPIO.output(LED_PIN, GPIO.HIGH)
            print("Weed detected! LED ON.")
        else:
            GPIO.output(LED_PIN, GPIO.LOW)
            print("No weeds detected. LED OFF.")

        # Show the annotated frame in a window
        annotated_frame = results[0].plot()  # Annotate frame with bounding boxes
        cv2.imshow("Weed Detection", annotated_frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting program.")

finally:
    # Cleanup GPIO and close windows
    GPIO.cleanup()
    cv2.destroyAllWindows()
