import os
import cv2
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
pyautogui.FAILSAFE = False

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

current_x = None
current_y = None

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
        print(gesture)
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

        print(result.hand_landmarks)

        if gesture.category_name == "Pointing_Up":
            first_hand = result.hand_landmarks[0]   # first hand
            index_tip = first_hand[8]               # index fingertip (landmark 8)

            # Convert normalized coordinates to pixels
            x_px = int(index_tip.x * frame.shape[1])
            y_px = int(index_tip.y * frame.shape[0])

            if current_x is None or current_y is None:
                current_x = x_px
                current_y = y_px
            else:
                dx = x_px - current_x
                dy = y_px - current_y

                # Move mouse cursor (scaled for sensitivity)
                scale = 2
                dx_scaled = int(dx * scale)
                dy_scaled = int(dy * scale)

                # Code to move the mouse cursor would go here
                pyautogui.moveRel(dx_scaled, dy_scaled, duration=0)

                current_x = x_px
                current_y = y_px

                cv2.circle(frame, (current_x, current_y), 10, (255, 0, 0), -1)
        else:
            current_x = None
            current_y = None


    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
