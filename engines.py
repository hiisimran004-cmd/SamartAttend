"""
engines.py — Combined face recognition + behavior detection engine
Merges face_engine.py + behavior_engine.py into a single import.

Face recognition: face_recognition library (dlib backend)
Behavior detection: OpenCV Haar cascades (no MediaPipe dependency)
"""

import os
import json
import logging
import cv2
import numpy as np
import face_recognition

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# FACE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def load_encodings(subject: str) -> dict:
    """
    Load face encodings for all students in a given subject/class.
    Returns: { rollNo: {"encoding": np.ndarray, "name": str} }
    """
    images_dir        = os.path.join("assets", "class", subject, "images")
    student_data_path = os.path.join("assets", "class", subject, "studentdata.json")

    if not os.path.isdir(images_dir):
        logger.warning(f"Images directory not found: {images_dir}")
        return {}

    name_to_roll = {}
    name_map     = {}
    if os.path.exists(student_data_path):
        with open(student_data_path, "r") as f:
            students = json.load(f)
        for s in students:
            key = s["name"].lower().replace(" ", "_")
            name_to_roll[key] = s["rollNo"]
            name_map[key]     = s["name"]

    encodings = {}
    supported = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(images_dir):
        if not filename.lower().endswith(supported):
            continue

        image_path = os.path.join(images_dir, filename)
        base       = os.path.splitext(filename)[0].lower().replace(" ", "_")

        try:
            img       = face_recognition.load_image_file(image_path)
            face_encs = face_recognition.face_encodings(img, num_jitters=0)

            if not face_encs:
                logger.warning(f"No face found in {filename}, skipping.")
                continue

            roll_no      = name_to_roll.get(base, base)
            display_name = name_map.get(base, base)

            encodings[roll_no] = {
                "encoding": face_encs[0],
                "name":     display_name
            }
            logger.info(f"Encoded: {display_name} ({roll_no})")

        except Exception as e:
            logger.error(f"Error encoding {filename}: {e}")

    logger.info(f"Total encodings for '{subject}': {len(encodings)}")
    return encodings


def identify_faces(frame: np.ndarray, encodings: dict) -> list:
    """
    Identify faces in a video frame.
    Returns: [{ "rollNo": str, "name": str, "location": tuple }]
    """
    if not encodings:
        return []

    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1], dtype=np.uint8)

    face_locations = face_recognition.face_locations(
        rgb_frame, number_of_times_to_upsample=1, model="hog"
    )
    if not face_locations:
        return []

    try:
        frame_encodings = face_recognition.face_encodings(
            rgb_frame, known_face_locations=face_locations, num_jitters=0
        )
    except TypeError:
        frame_encodings = face_recognition.face_encodings(rgb_frame, num_jitters=0)

    if not frame_encodings:
        return []

    known_rolls = list(encodings.keys())
    known_encs  = [encodings[r]["encoding"] for r in known_rolls]

    identified = []
    for face_enc, face_loc in zip(frame_encodings, face_locations):
        try:
            matches        = face_recognition.compare_faces(known_encs, face_enc, tolerance=0.55)
            face_distances = face_recognition.face_distance(known_encs, face_enc)

            if not any(matches):
                continue

            best_idx = int(np.argmin(face_distances))
            if matches[best_idx]:
                roll_no = known_rolls[best_idx]
                identified.append({
                    "rollNo":   roll_no,
                    "name":     encodings[roll_no]["name"],
                    "location": face_loc
                })
        except Exception as e:
            logger.error(f"Face matching error: {e}")

    return identified


# ═══════════════════════════════════════════════════════════════════════════
# BEHAVIOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════

# Load Haar cascades once (module-level for performance)
_eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def analyze_behavior(frame: np.ndarray, face_location: tuple) -> dict:
    """
    Analyze student behavior from face bounding box using OpenCV only.
    Returns: {"behavior": "focused"|"distracted"|"phone", "yaw": float, "pitch": float}
    """
    if frame is None or frame.size == 0:
        return {"behavior": "focused", "yaw": 0.0, "pitch": 0.0}

    top, right, bottom, left = face_location
    h, w = frame.shape[:2]

    top    = max(0, top);    left   = max(0, left)
    bottom = min(h, bottom); right  = min(w, right)

    if bottom <= top or right <= left:
        return {"behavior": "focused", "yaw": 0.0, "pitch": 0.0}

    face_w  = right - left
    face_h  = bottom - top
    face_roi  = frame[top:bottom, left:right]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    eyes = _eye_cascade.detectMultiScale(
        gray_face, scaleFactor=1.1, minNeighbors=3,
        minSize=(int(face_w * 0.1), int(face_h * 0.1))
    )

    yaw   = 0.0
    pitch = 0.0

    if len(eyes) >= 2:
        eyes      = sorted(eyes, key=lambda e: e[0])
        left_eye  = eyes[0]
        right_eye = eyes[1]

        lx = left_eye[0]  + left_eye[2]  // 2
        rx = right_eye[0] + right_eye[2] // 2
        ly = left_eye[1]  + left_eye[3]  // 2
        ry = right_eye[1] + right_eye[3] // 2

        face_center_x = face_w // 2
        eye_mid_x     = (lx + rx) // 2
        eye_mid_y     = (ly + ry) // 2

        yaw_ratio = (eye_mid_x - face_center_x) / (face_w / 2 + 1e-5)
        yaw       = yaw_ratio * 45.0

        eye_y_ratio = eye_mid_y / (face_h + 1e-5)
        pitch       = (0.40 - eye_y_ratio) * 100.0

        eye_dist  = rx - lx
        eye_ratio = eye_dist / (face_w + 1e-5)
        if eye_ratio < 0.25:
            yaw = 35.0

    elif len(eyes) == 1:
        yaw = 30.0
    else:
        faces_in_roi = _face_cascade.detectMultiScale(gray_face, 1.1, 3)
        if len(faces_in_roi) == 0:
            pitch = -35.0
        else:
            yaw = 28.0

    # Phone detection
    phone_detected = _detect_phone_contour(frame, face_location, w, h)

    abs_yaw = abs(yaw)
    if pitch < -30 or phone_detected:
        behavior = "phone"
    elif abs_yaw > 25:
        behavior = "distracted"
    elif abs_yaw < 15 and pitch > -20:
        behavior = "focused"
    else:
        behavior = "distracted"

    return {
        "behavior": behavior,
        "yaw":      round(yaw, 2),
        "pitch":    round(pitch, 2)
    }


def _detect_phone_contour(frame: np.ndarray, face_location: tuple,
                           frame_w: int, frame_h: int) -> bool:
    """Detect rectangular phone-shaped object near/below face."""
    top, right, bottom, left = face_location
    face_area = (bottom - top) * (right - left)

    roi_top    = max(0,       bottom - int((bottom - top) * 0.3))
    roi_bottom = min(frame_h, bottom + (bottom - top))
    roi_left   = max(0,       left   - 30)
    roi_right  = min(frame_w, right  + 30)

    if roi_top >= roi_bottom or roi_left >= roi_right:
        return False

    roi   = frame[roi_top:roi_bottom, roi_left:roi_right]
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if face_area * 0.3 < area < face_area * 2.0:
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, cw, ch = cv2.boundingRect(approx)
                if ch / (cw + 1e-5) > 1.2:
                    return True
    return False


def calculate_focus_score(behavior_samples: list) -> int:
    """
    Calculate focus score (0-100) from list of behavior strings.
    Weighted: focused=100pts, distracted=40pts, phone=0pts.
    """
    if not behavior_samples:
        return 0
    weights = {"focused": 100, "distracted": 40, "phone": 0}
    total = sum(weights.get(b, 50) for b in behavior_samples)
    return int(total / len(behavior_samples))