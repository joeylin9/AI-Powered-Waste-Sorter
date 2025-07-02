from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import tensorflow as tf
from keras.layers import DepthwiseConv2D
import time
import serial

# Adjust the COM port to match your system (e.g., 'COM3' for Windows or '/dev/ttyUSB0' on Linux/Mac)
arduino = serial.Serial('COM3', 9600)
time.sleep(2)  # Wait for Arduino to initialize


class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)  # Remove problematic parameter
        super().__init__(**kwargs)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
# model = load_model("keras_Model.h5", compile=False)
model = tf.keras.models.load_model(
    "keras_Model.h5", 
    custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}, compile=False
    )

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

current_time = time.time()
while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    if time.time() - current_time > 1:
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        current_time = time.time()
    
    # Get predicted label without newline/extra characters
    label = class_name.strip().lower()[2:]

    # Map label to signal character
    label_to_signal = {
        "plastic": b'P',
        "metal": b'M',
        "paper": b'A',
    }

    # Send to Arduino if label matches
    if label in label_to_signal:
        arduino.write(label_to_signal[label])
    else:
        arduino.write(b'X')

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

arduino.close()
camera.release()
cv2.destroyAllWindows()