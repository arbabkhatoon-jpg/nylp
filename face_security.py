"""
Face Recognition Security System
--------------------------------
Owner ka face pehchanta hai. Koi aur samne aye to RED ALERT dikhata hai,
beep bajata hai aur intruder ki photo save kar leta hai.

Install:
    pip install opencv-contrib-python numpy

Use (ye 3 steps ek ek karke chalao):
    python face_security.py register      # owner ka face capture
    python face_security.py train         # model train
    python face_security.py run           # security system start

Keys (run mode me):  q = quit
"""

import argparse
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

try:
    import winsound  # Windows only
except ImportError:
    winsound = None

# ---------------- Settings ----------------
DATASET_DIR = "dataset/owner"
INTRUDER_DIR = "intruders"
MODEL_FILE = "trainer.yml"
CAMERA_INDEX = 0
SAMPLES_TO_CAPTURE = 80
FACE_SIZE = (200, 200)

# LBPH me confidence jitni KAM ho, match utna behtar hota hai.
# Agar owner ko bhi unauthorized bol raha hai to is value ko barhao (e.g. 80-90).
# Agar koi aur banda owner ban jata hai to isko kam karo (e.g. 50-60).
THRESHOLD = 70

ALERT_BEEP_COOLDOWN = 1.5     # seconds
SNAPSHOT_COOLDOWN = 5         # seconds
# ------------------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera open nahi ho raha. CAMERA_INDEX change karke dekho (0, 1, 2).")
        sys.exit(1)
    return cap


def detect_faces(gray):
    return face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=6, minSize=(90, 90)
    )


def prepare_face(gray, box):
    x, y, w, h = box
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, FACE_SIZE)
    return cv2.equalizeHist(face)


def beep():
    def _beep():
        if winsound:
            winsound.Beep(1200, 300)
        else:
            print("\a", end="", flush=True)
    threading.Thread(target=_beep, daemon=True).start()


# ---------------- 1. Register ----------------
def register():
    os.makedirs(DATASET_DIR, exist_ok=True)
    cap = open_camera()
    count = 0
    last_save = 0
    print("Camera ke samne dekho. Thora sar idhar udhar ghumao, light alag alag rakho.")

    while count < SAMPLES_TO_CAPTURE:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if time.time() - last_save > 0.15:
                face = prepare_face(gray, faces[0])
                cv2.imwrite(os.path.join(DATASET_DIR, f"owner_{count:03d}.jpg"), face)
                count += 1
                last_save = time.time()
        elif len(faces) > 1:
            cv2.putText(frame, "Sirf ek banda camera me ho", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Captured: {count}/{SAMPLES_TO_CAPTURE}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Register Owner (q = stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {count} images save hui. Ab 'python face_security.py train' chalao.")


# ---------------- 2. Train ----------------
def train():
    if not os.path.isdir(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print("Pehle 'register' chalao.")
        sys.exit(1)

    images, labels = [], []
    for name in os.listdir(DATASET_DIR):
        img = cv2.imread(os.path.join(DATASET_DIR, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)
        labels.append(0)  # 0 = owner

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    recognizer.write(MODEL_FILE)
    print(f"Model train ho gaya ({len(images)} images). Ab 'python face_security.py run' chalao.")


# ---------------- 3. Run ----------------
def run():
    if not os.path.isfile(MODEL_FILE):
        print("Model nahi mila. Pehle 'register' aur 'train' chalao.")
        sys.exit(1)

    os.makedirs(INTRUDER_DIR, exist_ok=True)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    cap = open_camera()
    last_beep = 0
    last_snapshot = 0
    print("Security system chal raha hai. Band karne ke liye 'q' dabao.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        intruder = False

        for box in faces:
            x, y, w, h = box
            face = prepare_face(gray, box)
            _, confidence = recognizer.predict(face)

            if confidence < THRESHOLD:
                color = (0, 200, 0)
                text = f"OWNER ({confidence:.0f})"
            else:
                color = (0, 0, 255)
                text = f"UNAUTHORIZED ({confidence:.0f})"
                intruder = True

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        now = time.time()
        h_frame, w_frame = frame.shape[:2]

        if intruder:
            # Flashing red overlay
            if int(now * 4) % 2 == 0:
                red = np.full_like(frame, (0, 0, 255))
                frame = cv2.addWeighted(frame, 0.6, red, 0.4, 0)
            cv2.rectangle(frame, (0, 0), (w_frame - 1, h_frame - 1), (0, 0, 255), 12)
            cv2.rectangle(frame, (0, 0), (w_frame, 55), (0, 0, 180), -1)
            cv2.putText(frame, "ALERT! UNAUTHORIZED PERSON", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            if now - last_beep > ALERT_BEEP_COOLDOWN:
                beep()
                last_beep = now

            if now - last_snapshot > SNAPSHOT_COOLDOWN:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(INTRUDER_DIR, f"intruder_{stamp}.jpg"), frame)
                print(f"[{stamp}] Unauthorized person detected - photo saved.")
                last_snapshot = now
        elif len(faces) > 0:
            cv2.putText(frame, "ACCESS GRANTED", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3)
        else:
            cv2.putText(frame, "Monitoring...", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Face Security System (q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    modes = {"register": register, "train": train, "run": run}

    # Command line se mode mile to wo use hoga (python face_security.py run),
    # warna (Spyder me) console me puch lega.
    if len(sys.argv) > 1 and sys.argv[1] in modes:
        mode = sys.argv[1]
    else:
        mode = ""
        while mode not in modes:
            mode = input("Mode likho (register / train / run): ").strip().lower()

    modes[mode]()
