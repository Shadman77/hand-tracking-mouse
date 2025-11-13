import os
import cv2
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# 🗂️ Step 1. Ensure model exists
# -----------------------------
MODEL_DIR = "data"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_recognizer.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/gesture_recognizer.task"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"📁 Created folder: {MODEL_DIR}")

if not os.path.exists(MODEL_PATH):
    print("⬇️  Downloading gesture recognition model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded successfully!")

# -----------------------------
# ✋ Step 2. Load the model
# -----------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# -----------------------------
# 📷 Step 3. Start webcam feed
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    # Run recognition
    result = recognizer.recognize(mp_image)

    # Display result
    if result.gestures:
        gesture = result.gestures[0][0]
        name = gesture.category_name
        score = gesture.score
        cv2.putText(
            frame,
            f"{name} ({score:.2f})",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
