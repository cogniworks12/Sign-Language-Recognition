from flask import Flask, Response, render_template_string, jsonify
import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
import time
import threading

app = Flask(__name__)

# Load trained model
trained_model = tf.keras.models.load_model("my_model.keras")

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Sign labels
categories = {0: "A", 1: "C", 2: "G", 3: "I", 4: "K", 5: "N", 6: "O", 7: "R", 8: "S", 9: "W"}

# Image size for preprocessing
image_size = 128

# Global variables
recognized_text = ""  # Stores full detected text
current_sign = ""  # Currently detected sign
last_detected_sign = None  # Store last sign
sign_start_time = time.time()  # Timer to check if sign is stable
frame_lock = threading.Lock()  # Prevent threading issues

# OpenCV video capture in a separate thread
cap = cv2.VideoCapture(0)


def preprocess_image(image):
    """Preprocess image for model prediction."""
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_gray, (image_size, image_size))
    image_expanded = np.expand_dims(image_resized, axis=-1)  # Add channel dimension
    image_expanded = np.expand_dims(image_expanded, axis=0)  # Add batch dimension
    image_normalized = image_expanded.astype('float32') / 255.0
    return image_normalized


def process_frame():
    """Capture frame, detect hand sign, and update recognized text."""
    global recognized_text, current_sign, last_detected_sign, sign_start_time

    with frame_lock:
        success, frame = cap.read()
        if not success:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_sign = None  # Reset detected sign

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                frame_to_predict = frame.copy()
                h, w, c = frame.shape
                x_min, y_min, x_max, y_max = w, h, 0, 0

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                padding = 10
                x_min, y_min = max(x_min - padding, 0), max(y_min - padding, 0)
                x_max, y_max = min(x_max + padding, w), min(y_max + padding, h)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                hand_roi = frame_to_predict[y_min:y_max, x_min:x_max]
                if hand_roi.size > 0:
                    preprocessed_hand = preprocess_image(hand_roi)
                    predictions = trained_model.predict(preprocessed_hand)
                    predicted_class = np.argmax(predictions)

                    if predicted_class in categories:
                        detected_sign = categories[predicted_class]

        # Update the displayed current sign
        if detected_sign:
            current_sign = detected_sign

            # If new sign detected, reset timer
            if detected_sign != last_detected_sign:
                last_detected_sign = detected_sign
                sign_start_time = time.time()

            # If the sign remains stable for 3 seconds, add it to the recognized text
            if time.time() - sign_start_time >= 3:
                if len(recognized_text) < 50:  # Allow up to 50 characters
                    recognized_text += detected_sign
                sign_start_time = time.time()  # Reset timer

        # Display text on the frame
        cv2.putText(frame, f"Detected Sign: {current_sign}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Word: {recognized_text}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return frame


def gen_frames():
    """Generate real-time video stream."""
    while True:
        frame = process_frame()
        if frame is None:
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Render the HTML template."""
    return render_template_string(html_template)


@app.route('/video_feed')
def video_feed():
    """Video stream route."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_text')
def get_text():
    """Return the recognized text dynamically."""
    return jsonify({"recognized_text": recognized_text, "current_sign": current_sign})


# HTML Template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Gesture Recognition</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin: 50px; }
        .container { display: flex; flex-direction: column; align-items: center; }
        img { width: 80%; max-width: 800px; }
        #recognized-text, #current-sign {
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        }
        #recognized-text { color: #007bff; }
        #current-sign { color: #e74c3c; }
    </style>
</head>
<body>
    <h1>Real-Time Gesture Recognition</h1>
    <div class="container">
        <img src="{{ url_for('video_feed') }}" />
        <p id="current-sign">Current Sign: </p>
        <p id="recognized-text">Formed Word: </p>
    </div>

    <script>
        function updateRecognizedText() {
            fetch("/get_text")
                .then(response => response.json())
                .then(data => {
                    document.getElementById("current-sign").innerText = "Current Sign: " + data.current_sign;
                    document.getElementById("recognized-text").innerText = "Formed Word: " + data.recognized_text;
                })
                .catch(error => console.error("Error fetching recognized text:", error));
        }

        // Update detected text every 300ms
        setInterval(updateRecognizedText, 300);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
