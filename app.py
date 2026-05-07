"""
app.py — SmartAttend Flask Application (Refactored)
Changes:
  - Class schedule enforced: session only allowed during startTime–endTime window
  - Teacher name added to class creation & display
  - Attendance window logic: attendanceDelay opens window for windowDuration minutes only
    Students present ONLY during that window are eligible. Rest → absent.
  - 70% class time presence required to be marked 'present'
  - /scanner/live-data endpoint for admin monitor
  - PDF now actually downloads (send_file fix)
  - email_service.py imported correctly
"""

import os
import json
import base64
import uuid
import logging
from datetime import datetime, timezone, timedelta
from functools import wraps

import cv2
import numpy as np
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, send_from_directory, send_file)
from flask_cors import CORS

from firebase_config import db
from engines import load_encodings, identify_faces, analyze_behavior, calculate_focus_score
from email_service import generate_pdf, send_email, check_and_send_alerts, _build_performance_data

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "smartattend-secret-2024")
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ADMIN_PIN = "1234"

_encodings_cache: dict = {}

# Active session state
_active_session: dict = {
    "sessionId":        None,
    "classId":          None,
    "subject":          None,
    "chunkData":        {},
    "classDuration":    50,
    # NEW: attendance window tracking
    "attendanceWindowOpen":   False,
    "attendanceWindowStart":  None,   # datetime when window opened
    "attendanceWindowEnd":    None,   # datetime when window closes
    "windowEligible":         set(),  # rollNos seen during window
    # live data for monitor
    "liveDetections":         {},     # rollNo → {name, behavior, lastSeen}
    # schedule times for countdown
    "classEndTime":           None,   # datetime of class end
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


def decode_base64_frame(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _get_encodings(subject: str) -> dict:
    if subject not in _encodings_cache:
        _encodings_cache[subject] = load_encodings(subject)
    return _encodings_cache[subject]


def _ensure_class_folder(subject: str):
    base = os.path.join("assets", "class", subject)
    os.makedirs(os.path.join(base, "images"), exist_ok=True)
    data_file = os.path.join(base, "studentdata.json")
    if not os.path.exists(data_file):
        with open(data_file, "w") as f:
            json.dump([], f)


def _load_student_data(subject: str) -> list:
    path = os.path.join("assets", "class", subject, "studentdata.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_student_data(subject: str, data: list):
    path = os.path.join("assets", "class", subject, "studentdata.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _calculate_final_status(present_chunks: int, total_chunks: int, was_window_eligible: bool) -> str:
    """
    New logic:
    - Student must be window-eligible (seen during attendance window)
    - Then must be present for ≥70% of class chunks to be 'present'
    - If not window-eligible → 'absent' regardless
    - If window-eligible but <70% → 'late' if ≥30%, else 'absent'
    """
    if not was_window_eligible:
        return "absent"
    if total_chunks == 0:
        return "absent"
    ratio = present_chunks / total_chunks
    if ratio >= 0.70:
        return "present"
    elif ratio >= 0.30:
        return "late"
    return "absent"


def _dominant_behavior(samples: list) -> str:
    if not samples:
        return "unknown"
    counts = {}
    for b in samples:
        counts[b] = counts.get(b, 0) + 1
    return max(counts, key=counts.get)


def _parse_time_str(time_str: str) -> datetime:
    """Parse HH:MM time string to today's datetime."""
    h, m = map(int, time_str.split(":"))
    now = datetime.now()
    return now.replace(hour=h, minute=m, second=0, microsecond=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin():
    if not session.get("admin_logged_in"):
        return redirect(url_for("index"))
    return render_template("admin.html")


@app.route("/scanner")
def scanner():
    return render_template("scanner.html")


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json() or {}
    if data.get("pin", "") == ADMIN_PIN:
        session["admin_logged_in"] = True
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Invalid PIN"}), 401


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.clear()
    return jsonify({"success": True})


# ─────────────────────────────────────────────────────────────────────────────
# Class Management
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/admin/add-class", methods=["POST"])
@admin_required
def add_class():
    data    = request.get_json() or {}
    subject = data.get("subject", "").strip()
    room    = data.get("room", "").strip()
    teacher = data.get("teacher", "").strip()   # NEW: teacher name

    if not subject:
        return jsonify({"error": "Subject name is required"}), 400

    _ensure_class_folder(subject)
    class_id = f"{subject.lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"

    db.collection("classes").document(class_id).set({
        "subject":   subject,
        "room":      room,
        "teacher":   teacher,       # NEW
        "createdAt": datetime.now(timezone.utc)
    })

    logger.info(f"Class created: {subject} (ID: {class_id})")
    return jsonify({"success": True, "classId": class_id})


@app.route("/admin/classes", methods=["GET"])
@admin_required
def get_classes():
    classes = []
    for doc in db.collection("classes").stream():
        c = doc.to_dict()
        c["classId"] = doc.id
        classes.append(c)
    return jsonify(classes)


# ─────────────────────────────────────────────────────────────────────────────
# Student Management
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/admin/add-student", methods=["POST"])
@admin_required
def add_student():
    class_id     = request.form.get("classId", "")
    subject      = request.form.get("subject", "")
    name         = request.form.get("name", "").strip()
    roll_no      = request.form.get("rollNo", "").strip()
    phone        = request.form.get("phone", "")
    email        = request.form.get("email", "")
    parent_email = request.form.get("parentEmail", "")

    if not all([class_id, subject, name, roll_no]):
        return jsonify({"error": "classId, subject, name, rollNo required"}), 400

    image_file = request.files.get("image")
    image_path = ""
    if image_file:
        _ensure_class_folder(subject)
        safe_name  = name.lower().replace(" ", "_")
        filename   = f"{safe_name}.jpg"
        image_path = os.path.join("assets", "class", subject, "images", filename)
        image_file.save(image_path)

    stu_data = {
        "name":        name,
        "phone":       phone,
        "email":       email,
        "parentEmail": parent_email,
        "imagePath":   image_path,
        "enrolledAt":  datetime.now(timezone.utc)
    }
    db.collection("classes").document(class_id) \
      .collection("students").document(roll_no).set(stu_data)

    students = _load_student_data(subject)
    students = [s for s in students if s.get("rollNo") != roll_no]
    students.append({"rollNo": roll_no, **stu_data, "enrolledAt": str(datetime.now())})
    _save_student_data(subject, students)

    _encodings_cache.pop(subject, None)
    logger.info(f"Student added: {name} ({roll_no}) to {subject}")
    return jsonify({"success": True})


@app.route("/admin/students/<class_id>", methods=["GET"])
@admin_required
def get_students(class_id):
    students = []
    for doc in db.collection("classes").document(class_id) \
                 .collection("students").stream():
        s = doc.to_dict()
        s["rollNo"] = doc.id
        students.append(s)
    return jsonify(students)


# ─────────────────────────────────────────────────────────────────────────────
# Schedule Management
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/admin/set-schedule", methods=["POST"])
@admin_required
def set_schedule():
    data = request.get_json() or {}
    class_id = data.get("classId")
    if not class_id:
        return jsonify({"error": "classId required"}), 400

    schedule = {
        "days":              data.get("days", []),
        "startTime":         data.get("startTime", ""),
        "endTime":           data.get("endTime", ""),
        "attendanceDelay":   int(data.get("attendanceDelay", 5)),
        "windowDuration":    int(data.get("windowDuration", 1)),    # NEW: how long window stays open (minutes)
        "chunkInterval":     int(data.get("chunkInterval", 5)),
    }
    db.collection("classes").document(class_id) \
      .collection("schedule").document("config").set(schedule)

    return jsonify({"success": True})


@app.route("/admin/get-schedule/<class_id>", methods=["GET"])
@admin_required
def get_schedule(class_id):
    doc = db.collection("classes").document(class_id) \
             .collection("schedule").document("config").get()
    if doc.exists:
        return jsonify(doc.to_dict())
    return jsonify({})


# ─────────────────────────────────────────────────────────────────────────────
# Session Management
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/admin/start-session", methods=["POST"])
@admin_required
def start_session():
    data     = request.get_json() or {}
    class_id = data.get("classId", "")
    subject  = data.get("subject", "")

    if not class_id:
        return jsonify({"error": "classId required"}), 400

    # ── Fetch schedule ──────────────────────────────────────────────────
    sched_doc = db.collection("classes").document(class_id) \
                  .collection("schedule").document("config").get()
    schedule  = sched_doc.to_dict() if sched_doc.exists else {}

    start_time_str   = schedule.get("startTime", "")
    end_time_str     = schedule.get("endTime", "")
    attendance_delay = int(schedule.get("attendanceDelay", 5))
    window_duration  = int(schedule.get("windowDuration", 1))
    chunk_interval   = int(schedule.get("chunkInterval", 5))

    # ── Schedule enforcement: only allow during class time ──────────────
    now = datetime.now()
    class_start_dt = None
    class_end_dt   = None

    if start_time_str and end_time_str:
        try:
            class_start_dt = _parse_time_str(start_time_str)
            class_end_dt   = _parse_time_str(end_time_str)
            # Allow starting 10 seconds before class
            if now < class_start_dt - timedelta(seconds=10):
                secs_until = int((class_start_dt - now).total_seconds())
                return jsonify({
                    "error": f"Class starts at {start_time_str}. Wait {secs_until}s.",
                    "secsUntilStart": secs_until
                }), 400
            if now > class_end_dt:
                return jsonify({"error": f"Class ended at {end_time_str}. Too late to start."}), 400
        except Exception as e:
            logger.warning(f"Schedule parse error: {e}")

    # ── Calculate class duration ────────────────────────────────────────
    class_duration = 50
    if start_time_str and end_time_str:
        try:
            sh, sm = map(int, start_time_str.split(":"))
            eh, em = map(int, end_time_str.split(":"))
            class_duration = (eh * 60 + em) - (sh * 60 + sm)
            if class_duration <= 0:
                class_duration = 50
        except Exception:
            pass

    chunk_times = list(range(attendance_delay, class_duration + 1, chunk_interval))

    # ── Attendance window timing ────────────────────────────────────────
    # Window opens at: class_start + attendanceDelay minutes
    # Window closes at: window_open + windowDuration minutes
    window_open_dt  = None
    window_close_dt = None
    if class_start_dt:
        window_open_dt  = class_start_dt + timedelta(minutes=attendance_delay)
        window_close_dt = window_open_dt + timedelta(minutes=window_duration)
    else:
        # fallback: window opens now + delay, closes + windowDuration
        window_open_dt  = now + timedelta(minutes=attendance_delay)
        window_close_dt = window_open_dt + timedelta(minutes=window_duration)

    session_id = str(uuid.uuid4())
    today      = now.strftime("%Y-%m-%d")

    db.collection("classes").document(class_id) \
      .collection("sessions").document(session_id).set({
          "date":            today,
          "startedAt":       datetime.now(timezone.utc),
          "status":          "active",
          "classId":         class_id,
          "chunkTimes":      chunk_times,
          "classDuration":   class_duration,
          "windowOpenAt":    window_open_dt.isoformat(),
          "windowCloseAt":   window_close_dt.isoformat(),
          "attendanceDelay": attendance_delay,
          "windowDuration":  window_duration,
      })

    _active_session.update({
        "sessionId":              session_id,
        "classId":                class_id,
        "subject":                subject,
        "chunkData":              {},
        "classDuration":          class_duration,
        "attendanceWindowOpen":   False,
        "attendanceWindowStart":  window_open_dt,
        "attendanceWindowEnd":    window_close_dt,
        "windowEligible":         set(),
        "liveDetections":         {},
        "classEndTime":           class_end_dt,
    })

    _encodings_cache[subject] = load_encodings(subject)

    logger.info(f"Session started: {session_id} | Duration: {class_duration}min | Window: {window_open_dt} – {window_close_dt}")

    # Seconds until class end (for countdown)
    secs_remaining = None
    if class_end_dt:
        secs_remaining = max(0, int((class_end_dt - now).total_seconds()))

    return jsonify({
        "success":          True,
        "sessionId":        session_id,
        "chunkTimes":       chunk_times,
        "classDuration":    class_duration,
        "secsRemaining":    secs_remaining,
        "windowOpenISO":    window_open_dt.isoformat(),
        "windowCloseISO":   window_close_dt.isoformat(),
        "startTimeStr":     start_time_str,
        "endTimeStr":       end_time_str,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Scanner Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/scanner/frame", methods=["POST"])
def process_frame():
    data     = request.get_json() or {}
    b64frame = data.get("frame", "")
    if not b64frame:
        return jsonify({"error": "No frame data"}), 400

    subject = _active_session.get("subject")
    if not subject:
        return jsonify({"error": "No active session"}), 400

    frame = decode_base64_frame(b64frame)
    if frame is None:
        return jsonify({"error": "Invalid frame"}), 400

    encodings  = _get_encodings(subject)
    identified = identify_faces(frame, encodings)

    # ── Check if attendance window is currently open ──────────────────
    now          = datetime.now()
    win_start    = _active_session.get("attendanceWindowStart")
    win_end      = _active_session.get("attendanceWindowEnd")
    window_open  = win_start and win_end and (win_start <= now <= win_end)
    _active_session["attendanceWindowOpen"] = window_open

    results = []
    for person in identified:
        roll_no  = person["rollNo"]
        face_loc = person["location"]

        beh_result = analyze_behavior(frame, face_loc)
        behavior   = beh_result["behavior"]

        # Accumulate chunk data
        if roll_no not in _active_session["chunkData"]:
            _active_session["chunkData"][roll_no] = {
                "behavior_samples": [],
                "present":          True,
                "present_chunks":   0,
                "total_chunks":     0,
            }
        _active_session["chunkData"][roll_no]["behavior_samples"].append(behavior)
        _active_session["chunkData"][roll_no]["present"] = True

        # Mark window-eligible
        if window_open:
            _active_session["windowEligible"].add(roll_no)

        # Update live detections
        _active_session["liveDetections"][roll_no] = {
            "name":     person["name"],
            "behavior": behavior,
            "yaw":      beh_result["yaw"],
            "pitch":    beh_result["pitch"],
            "location": face_loc,
            "lastSeen": now.isoformat(),
        }

        results.append({
            "rollNo":    roll_no,
            "name":      person["name"],
            "behavior":  behavior,
            "yaw":       beh_result["yaw"],
            "pitch":     beh_result["pitch"],
            "location":  face_loc,
        })

    return jsonify({
        "detected":     results,
        "count":        len(results),
        "windowOpen":   window_open,
        "windowStart":  win_start.isoformat() if win_start else None,
        "windowEnd":    win_end.isoformat()   if win_end   else None,
    })


@app.route("/scanner/live-data", methods=["GET"])
def live_data():
    """Admin monitor polls this to get current detected students."""
    live = _active_session.get("liveDetections", {})
    session_id = _active_session.get("sessionId")
    class_end  = _active_session.get("classEndTime")
    now        = datetime.now()

    secs_remaining = None
    if class_end:
        secs_remaining = max(0, int((class_end - now).total_seconds()))

    # Window info
    win_start = _active_session.get("attendanceWindowStart")
    win_end   = _active_session.get("attendanceWindowEnd")
    window_open = win_start and win_end and (win_start <= now <= win_end)

    return jsonify({
        "active":       session_id is not None,
        "sessionId":    session_id,
        "subject":      _active_session.get("subject"),
        "detections":   list(live.values()),
        "secsRemaining": secs_remaining,
        "windowOpen":   window_open,
        "windowEnd":    win_end.isoformat() if win_end else None,
    })


@app.route("/scanner/save-chunk", methods=["POST"])
def save_chunk():
    data        = request.get_json() or {}
    chunk_index = data.get("chunkIndex", 0)

    session_id = _active_session.get("sessionId")
    class_id   = _active_session.get("classId")
    if not session_id or not class_id:
        return jsonify({"error": "No active session"}), 400

    chunk_data_ref = db.collection("classes").document(class_id) \
                       .collection("sessions").document(session_id) \
                       .collection("chunks").document(str(chunk_index))
    chunk_data_ref.set({"timestamp": datetime.now(timezone.utc)})

    chunk_summary = {}
    for roll_no, cd in _active_session["chunkData"].items():
        samples   = cd["behavior_samples"]
        focus_scr = calculate_focus_score(samples)
        dom_beh   = _dominant_behavior(samples)

        chunk_data_ref.collection("attendance").document(roll_no).set({
            "present":    cd["present"],
            "behavior":   dom_beh,
            "focusScore": focus_scr
        })
        chunk_summary[roll_no] = {"present": cd["present"], "behavior": dom_beh, "focusScore": focus_scr}

        # Track chunk presence counts
        _active_session["chunkData"][roll_no]["total_chunks"] = \
            _active_session["chunkData"][roll_no].get("total_chunks", 0) + 1
        if cd["present"]:
            _active_session["chunkData"][roll_no]["present_chunks"] = \
                _active_session["chunkData"][roll_no].get("present_chunks", 0) + 1

    # Reset for next chunk
    for roll_no in _active_session["chunkData"]:
        _active_session["chunkData"][roll_no]["behavior_samples"] = []
        _active_session["chunkData"][roll_no]["present"] = False

    logger.info(f"Chunk {chunk_index} saved")
    return jsonify({"success": True, "chunk": chunk_index, "data": chunk_summary})


@app.route("/scanner/end-session", methods=["POST"])
def end_session():
    session_id = _active_session.get("sessionId")
    class_id   = _active_session.get("classId")

    if not session_id or not class_id:
        return jsonify({"error": "No active session"}), 400

    # Save final chunk
    if _active_session.get("chunkData"):
        try:
            chunk_data_ref = db.collection("classes").document(class_id) \
                               .collection("sessions").document(session_id) \
                               .collection("chunks").document("final")
            chunk_data_ref.set({"timestamp": datetime.now(timezone.utc)})
            for roll_no, cd in _active_session["chunkData"].items():
                samples   = cd.get("behavior_samples", [])
                focus_scr = calculate_focus_score(samples)
                dom_beh   = _dominant_behavior(samples) if samples else "focused"
                chunk_data_ref.collection("attendance").document(roll_no).set({
                    "present":    True,
                    "behavior":   dom_beh,
                    "focusScore": focus_scr
                })
        except Exception as e:
            logger.error(f"Final chunk save error: {e}")

    # Fetch all chunks
    chunks_ref = db.collection("classes").document(class_id) \
                   .collection("sessions").document(session_id) \
                   .collection("chunks").stream()

    student_chunks = {}
    for chunk_doc in chunks_ref:
        for att_doc in chunk_doc.reference.collection("attendance").stream():
            rn  = att_doc.id
            att = att_doc.to_dict()
            if rn not in student_chunks:
                student_chunks[rn] = {
                    "present_count": 0,
                    "total_chunks":  0,
                    "focus_scores":  [],
                    "behaviors":     [],
                    "phone_used":    False,
                }
            student_chunks[rn]["total_chunks"] += 1
            if att.get("present", False):
                student_chunks[rn]["present_count"] += 1
            student_chunks[rn]["focus_scores"].append(att.get("focusScore", 0))
            beh = att.get("behavior", "focused")
            student_chunks[rn]["behaviors"].append(beh)
            if beh == "phone":
                student_chunks[rn]["phone_used"] = True

    window_eligible = _active_session.get("windowEligible", set())
    final_results   = {}
    session_ref     = db.collection("classes").document(class_id) \
                        .collection("sessions").document(session_id)

    for roll_no, stats in student_chunks.items():
        is_eligible = roll_no in window_eligible
        status      = _calculate_final_status(
            stats["present_count"], stats["total_chunks"], is_eligible
        )
        avg_focus = (sum(stats["focus_scores"]) / len(stats["focus_scores"])
                     if stats["focus_scores"] else 0)
        dom_beh   = _dominant_behavior(stats["behaviors"])

        result = {
            "status":           status,
            "avgFocusScore":    round(avg_focus),
            "dominantBehavior": dom_beh,
            "presentChunks":    stats["present_count"],
            "totalChunks":      stats["total_chunks"],
            "phoneUsed":        stats["phone_used"],
            "windowEligible":   is_eligible,
        }
        session_ref.collection("finalResults").document(roll_no).set(result)
        final_results[roll_no] = result

    # Mark all enrolled students absent if not seen
    try:
        total_chunks_in_session = max(
            (s["total_chunks"] for s in student_chunks.values()), default=1
        )
        for stu_doc in db.collection("classes").document(class_id) \
                         .collection("students").stream():
            roll_no = stu_doc.id
            if roll_no not in student_chunks:
                absent_result = {
                    "status":           "absent",
                    "avgFocusScore":    0,
                    "dominantBehavior": "unknown",
                    "presentChunks":    0,
                    "totalChunks":      total_chunks_in_session,
                    "phoneUsed":        False,
                    "windowEligible":   False,
                }
                session_ref.collection("finalResults").document(roll_no).set(absent_result)
                final_results[roll_no] = absent_result
    except Exception as e:
        logger.error(f"Absent marking error: {e}")

    # Save to student attendance subcollection
    today = datetime.now().strftime("%Y-%m-%d")
    for roll_no, result in final_results.items():
        try:
            db.collection("classes").document(class_id) \
              .collection("students").document(roll_no) \
              .collection("attendance").document(today).set({
                  "sessionId":        session_id,
                  "date":             today,
                  "status":           result["status"],
                  "avgFocusScore":    result["avgFocusScore"],
                  "dominantBehavior": result["dominantBehavior"],
                  "presentChunks":    result["presentChunks"],
                  "totalChunks":      result["totalChunks"],
                  "savedAt":          datetime.now(timezone.utc)
              })
        except Exception as e:
            logger.error(f"Attendance save error ({roll_no}): {e}")

    session_ref.update({"endedAt": datetime.now(timezone.utc), "status": "ended"})

    _active_session.update({
        "sessionId": None, "classId": None, "subject": None,
        "chunkData": {}, "classDuration": 50,
        "attendanceWindowOpen": False,
        "attendanceWindowStart": None, "attendanceWindowEnd": None,
        "windowEligible": set(), "liveDetections": {}, "classEndTime": None,
    })

    return jsonify({"success": True, "results": final_results, "totalStudents": len(final_results)})


# ─────────────────────────────────────────────────────────────────────────────
# Active Session Info
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/admin/active-session", methods=["GET"])
@admin_required
def active_session_info():
    sess     = _active_session
    win_start = sess.get("attendanceWindowStart")
    win_end   = sess.get("attendanceWindowEnd")
    class_end = sess.get("classEndTime")
    now       = datetime.now()

    secs_remaining = None
    if class_end:
        secs_remaining = max(0, int((class_end - now).total_seconds()))

    return jsonify({
        "sessionId":     sess.get("sessionId"),
        "classId":       sess.get("classId"),
        "subject":       sess.get("subject"),
        "active":        sess.get("sessionId") is not None,
        "secsRemaining": secs_remaining,
        "windowOpenISO": win_start.isoformat() if win_start else None,
        "windowCloseISO": win_end.isoformat()  if win_end   else None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/admin/report/<roll_no>", methods=["GET"])
@admin_required
def get_report(roll_no):
    class_id = request.args.get("classId")
    if not class_id:
        return jsonify({"error": "classId query param required"}), 400

    from firebase_admin import firestore as fs

    stu_doc = db.collection("classes").document(class_id) \
                .collection("students").document(roll_no).get()
    if not stu_doc.exists:
        return jsonify({"error": "Student not found"}), 404

    stu_data = stu_doc.to_dict()
    stu_data["rollNo"] = roll_no

    cls_doc  = db.collection("classes").document(class_id).get()
    cls_data = cls_doc.to_dict() if cls_doc.exists else {}

    sessions_ref = db.collection("classes").document(class_id) \
                     .collection("sessions") \
                     .order_by("date", direction=fs.Query.DESCENDING) \
                     .limit(14).stream()

    session_records = []
    for sess_doc in sessions_ref:
        sess  = sess_doc.to_dict()
        fr_doc = db.collection("classes").document(class_id) \
                   .collection("sessions").document(sess_doc.id) \
                   .collection("finalResults").document(roll_no).get()
        if fr_doc.exists:
            fr = fr_doc.to_dict()
            session_records.append({
                "date":             sess.get("date", ""),
                "status":           fr.get("status", "absent"),
                "focusScore":       fr.get("avgFocusScore", 0),
                "avgFocusScore":    fr.get("avgFocusScore", 0),
                "dominantBehavior": fr.get("dominantBehavior", "unknown"),
                "phoneUsed":        fr.get("phoneUsed", False),
                "presentChunks":    fr.get("presentChunks", 0),
                "totalChunks":      fr.get("totalChunks", 0),
            })

    perf = _build_performance_data(session_records)
    return jsonify({"student": stu_data, "class": cls_data, "performance": perf})


@app.route("/admin/send-report", methods=["POST"])
@admin_required
def send_report():
    import time as time_mod
    data     = request.get_json() or {}
    roll_no  = data.get("rollNo", "")
    class_id = data.get("classId", "")

    if not roll_no or not class_id:
        return jsonify({"error": "rollNo and classId required"}), 400

    from firebase_admin import firestore as fs

    stu_doc = db.collection("classes").document(class_id) \
                .collection("students").document(roll_no).get()
    if not stu_doc.exists:
        return jsonify({"error": "Student not found"}), 404

    stu_data = {**stu_doc.to_dict(), "rollNo": roll_no}
    cls_doc  = db.collection("classes").document(class_id).get()
    cls_data = cls_doc.to_dict() if cls_doc.exists else {}
    stu_data.update({
        "subject": cls_data.get("subject", ""),
        "room":    cls_data.get("room", ""),
        "teacher": cls_data.get("teacher", ""),
    })

    sessions_ref = db.collection("classes").document(class_id) \
                     .collection("sessions") \
                     .order_by("date", direction=fs.Query.DESCENDING) \
                     .limit(14).stream()

    session_records = []
    for sess_doc in sessions_ref:
        sess   = sess_doc.to_dict()
        fr_doc = db.collection("classes").document(class_id) \
                   .collection("sessions").document(sess_doc.id) \
                   .collection("finalResults").document(roll_no).get()
        if fr_doc.exists:
            fr = fr_doc.to_dict()
            session_records.append({
                "date":             sess.get("date", ""),
                "status":           fr.get("status", "absent"),
                "focusScore":       fr.get("avgFocusScore", 0),
                "avgFocusScore":    fr.get("avgFocusScore", 0),
                "dominantBehavior": fr.get("dominantBehavior", "unknown"),
                "phoneUsed":        fr.get("phoneUsed", False),
            })

    perf = _build_performance_data(session_records)

    for key in list(stu_data.keys()):
        if isinstance(stu_data[key], str):
            stu_data[key] = stu_data[key].replace("\u2014", "-").replace("\u2013", "-")

    os.makedirs("static", exist_ok=True)
    pdf_path = os.path.join("static", f"report_{roll_no}_{int(time_mod.time())}.pdf")

    try:
        generate_pdf(stu_data, perf, pdf_path)
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500

    # Send email if parent email exists
    parent_email = stu_data.get("parentEmail", "")
    if parent_email:
        body = (
            f"Dear Parent/Guardian,\n\n"
            f"Please find the performance report for {stu_data.get('name','Student')} ({roll_no}).\n\n"
            f"Attendance Rate: {perf['attendance_rate']}%\n"
            f"Average Focus Score: {perf['avg_focus_score']}/100\n"
            f"Performance Trend: {perf['trend'].upper()}\n\n"
            f"Regards,\nSmartAttend System"
        )
        send_email(parent_email, f"Performance Report – {stu_data.get('name','')}", body, pdf_path)

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f"Report_{roll_no}.pdf",
        mimetype="application/pdf"
    )


@app.route("/admin/check-alerts", methods=["GET"])
@admin_required
def check_alerts():
    new_alerts = check_and_send_alerts(db)
    all_alerts = []
    for doc in db.collection("alerts").stream():
        a = doc.to_dict()
        a["alertId"] = doc.id
        if hasattr(a.get("createdAt"), "isoformat"):
            a["createdAt"] = a["createdAt"].isoformat()
        all_alerts.append(a)
    return jsonify({"alerts": all_alerts, "newAlerts": new_alerts})


# ─────────────────────────────────────────────────────────────────────────────
# Static assets
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/assets/<path:filename>")
def serve_asset(filename):
    return send_from_directory("assets", filename)


if __name__ == "__main__":
    os.makedirs(os.path.join("assets", "icons"), exist_ok=True)
    os.makedirs(os.path.join("assets", "class"), exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)