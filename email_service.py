"""
email_service.py — SmartAttend Email & PDF Service
Fixed: teacher name in PDF, parent email sends actual attachment,
       Unicode safe string handling.
"""

import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.base      import MIMEBase
from email                import encoders
from datetime             import datetime, timedelta, timezone

from fpdf import FPDF

logger = logging.getLogger(__name__)

SMTP_HOST       = "smtp.gmail.com"
SMTP_PORT       = 587
SENDER_EMAIL    = os.environ.get("SMTP_EMAIL",    "auth.samad.sayed@gmail.com")
SENDER_PASSWORD = os.environ.get("SMTP_PASSWORD", "bzwq zlmm qhbe ocqb")

LOGO_PATH = os.path.join("assets", "icons", "unilogo.png")


def _safe(text: str) -> str:
    """Replace characters not supported by Helvetica with ASCII equivalents."""
    if not text:
        return ""
    return (str(text)
            .replace("\u2014", "-")
            .replace("\u2013", "-")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2022", "*")
            .replace("\u00b0", " deg")
            .replace("\u00e9", "e")
            .replace("\u00e0", "a")
            )


class AttendancePDF(FPDF):
    def __init__(self, student_name: str):
        super().__init__()
        self.student_name = student_name
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        if os.path.exists(LOGO_PATH):
            try:
                self.image(LOGO_PATH, x=10, y=8, w=30)
            except Exception:
                pass
        self.set_font("Helvetica", "B", 16)
        self.set_xy(0, 8)
        self.cell(0, 10, "SmartAttend - Student Performance Report", align="C", ln=True)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 5, f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", align="C", ln=True)
        self.ln(5)
        self.set_draw_color(0, 200, 180)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()} | Confidential - SmartAttend System", align="C")
        self.set_text_color(0, 0, 0)

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(20, 20, 40)
        self.set_text_color(0, 200, 180)
        self.cell(0, 8, _safe(f"  {title}"), ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def kv_row(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.cell(65, 7, _safe(key) + ":", border=0)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 7, _safe(str(value)), border=0, ln=True)

    def table_header(self, cols: list, widths: list):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 30, 60)
        self.set_text_color(255, 255, 255)
        for col, w in zip(cols, widths):
            self.cell(w, 7, _safe(col), border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(0, 0, 0)

    def table_row(self, values: list, widths: list, fill: bool = False):
        self.set_font("Helvetica", "", 9)
        if fill:
            self.set_fill_color(240, 240, 255)
        else:
            self.set_fill_color(255, 255, 255)
        for val, w in zip(values, widths):
            self.cell(w, 6, _safe(str(val)), border=1, fill=fill, align="C")
        self.ln()


def generate_pdf(student_data: dict, performance_data: dict, output_path: str) -> str:
    for key in list(student_data.keys()):
        if isinstance(student_data[key], str):
            student_data[key] = _safe(student_data[key])

    pdf = AttendancePDF(student_data.get("name", "Unknown"))
    pdf.add_page()

    # Section 1: Student Info
    pdf.section_title("Student Information")
    pdf.kv_row("Name",         student_data.get("name", "N/A"))
    pdf.kv_row("Roll Number",  student_data.get("rollNo", "N/A"))
    pdf.kv_row("Email",        student_data.get("email", "N/A"))
    pdf.kv_row("Phone",        student_data.get("phone", "N/A"))
    pdf.kv_row("Subject",      student_data.get("subject", "N/A"))
    pdf.kv_row("Room",         student_data.get("room", "N/A"))
    pdf.kv_row("Teacher",      student_data.get("teacher", "N/A"))   # NEW
    pdf.ln(4)

    # Section 2: Attendance Summary
    pdf.section_title("Attendance Summary")
    att_rate   = performance_data.get("attendance_rate", 0)
    focus_avg  = performance_data.get("avg_focus_score", 0)
    perf_score = performance_data.get("performance_score", 0)
    trend      = performance_data.get("trend", "stable")

    pdf.kv_row("Attendance Rate",     f"{att_rate:.1f}%")
    pdf.kv_row("Avg Focus Score",     f"{focus_avg:.1f}/100")
    pdf.kv_row("Overall Performance", f"{perf_score:.1f}/100")
    pdf.kv_row("Trend",               trend.upper())
    pdf.ln(4)

    # Section 3: Behavior Breakdown
    pdf.section_title("Behavior Breakdown")
    beh = performance_data.get("behavior_breakdown", {})
    pdf.kv_row("Focused",    f"{beh.get('focused', 0):.1f}%")
    pdf.kv_row("Distracted", f"{beh.get('distracted', 0):.1f}%")
    pdf.kv_row("Phone Use",  f"{beh.get('phone', 0):.1f}%")
    pdf.ln(4)

    # Phone use alert
    sessions_list = performance_data.get("sessions", [])
    phone_sessions = [s for s in sessions_list if s.get("phoneUsed")]
    if phone_sessions:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(200, 50, 80)
        pdf.cell(0, 7, f"  ⚠ Phone detected in {len(phone_sessions)} session(s)", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    # Section 4: Session History
    sessions = performance_data.get("sessions", [])
    if sessions:
        pdf.section_title("Session History (Last 14 Sessions)")
        cols   = ["Date", "Status", "Focus", "Behavior", "Phone"]
        widths = [35, 28, 30, 52, 25]
        pdf.table_header(cols, widths)
        for i, s in enumerate(sessions[-14:]):
            phone_flag = "YES" if s.get("phoneUsed") else "no"
            pdf.table_row(
                [
                    s.get("date", "N/A"),
                    s.get("status", "N/A"),
                    str(s.get("focusScore", 0)),
                    s.get("dominantBehavior", "N/A"),
                    phone_flag,
                ],
                widths,
                fill=(i % 2 == 0)
            )
        pdf.ln(4)

    # Section 5: Suggestions
    pdf.section_title("Improvement Suggestions")
    suggestions = _generate_suggestions(performance_data)
    pdf.set_font("Helvetica", "", 10)
    for suggestion in suggestions:
        pdf.multi_cell(0, 6, _safe(f"  * {suggestion}"))
    pdf.ln(2)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    pdf.output(output_path)
    logger.info(f"PDF generated: {output_path}")
    return output_path


def _generate_suggestions(performance_data: dict) -> list:
    suggestions = []
    att   = performance_data.get("attendance_rate", 100)
    focus = performance_data.get("avg_focus_score", 100)
    beh   = performance_data.get("behavior_breakdown", {})
    trend = performance_data.get("trend", "stable")

    if att < 70:
        suggestions.append("CRITICAL: Attendance is below 70%. Immediate improvement required.")
    elif att < 80:
        suggestions.append("Attendance is below 80%. Regular class attendance is strongly advised.")
    elif att < 90:
        suggestions.append("Attendance is slightly below 90%. Try not to miss any sessions.")

    if focus < 40:
        suggestions.append("Focus score is critically low. Consider removing distractions and sitting front.")
    elif focus < 60:
        suggestions.append("Focus could be significantly improved. Engage actively with the lecture material.")
    elif focus < 75:
        suggestions.append("Focus score is moderate. Avoid phone use and stay engaged during class.")

    phone_pct = beh.get("phone", 0)
    if phone_pct > 30:
        suggestions.append(f"Phone use detected in {phone_pct:.0f}% of sessions. This seriously impacts performance.")
    elif phone_pct > 10:
        suggestions.append(f"Phone use in {phone_pct:.0f}% of sessions — please keep phone away during class.")

    dist_pct = beh.get("distracted", 0)
    if dist_pct > 50:
        suggestions.append(f"Distracted in {dist_pct:.0f}% of sessions. Consider reducing external distractions.")

    if trend == "critical":
        suggestions.append("CRITICAL DECLINE detected. Immediate meeting with instructor is recommended.")
    elif trend == "declining":
        suggestions.append("Performance is declining. Please seek academic support proactively.")
    elif trend == "improving":
        suggestions.append("Great progress! Performance is improving. Keep up the momentum.")

    if not suggestions:
        suggestions.append("Excellent! Attendance and focus are strong. Keep maintaining this performance.")

    return suggestions


def send_email(to_email: str, subject: str, body: str, attachment_path: str = None) -> bool:
    """Send email with optional PDF attachment."""
    try:
        msg = MIMEMultipart()
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{os.path.basename(attachment_path)}"'
            )
            msg.attach(part)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())

        logger.info(f"Email sent to {to_email}")
        return True

    except Exception as e:
        logger.error(f"Email failed to {to_email}: {e}")
        return False


def check_and_send_alerts(db) -> list:
    """
    Check all classes & students for:
    - 3+ consecutive absences → send alert email with PDF
    - 4+ consecutive declining sessions → flag alert
    Returns list of created alerts.
    """
    from firebase_admin import firestore as fs

    alerts_created = []
    classes_ref = db.collection("classes").stream()

    for cls_doc in classes_ref:
        class_id   = cls_doc.id
        class_data = cls_doc.to_dict()
        subject    = class_data.get("subject", class_id)
        room       = class_data.get("room", "")
        teacher    = class_data.get("teacher", "")

        students_ref = db.collection("classes").document(class_id) \
                         .collection("students").stream()

        for stu_doc in students_ref:
            roll_no      = stu_doc.id
            stu_data     = stu_doc.to_dict()
            parent_email = stu_data.get("parentEmail", "")
            name         = stu_data.get("name", roll_no)

            sessions_ref = db.collection("classes").document(class_id) \
                             .collection("sessions") \
                             .order_by("date", direction=fs.Query.DESCENDING) \
                             .limit(14).stream()

            session_records = []
            for sess in sessions_ref:
                sess_data = sess.to_dict()
                final_ref = db.collection("classes").document(class_id) \
                              .collection("sessions").document(sess.id) \
                              .collection("finalResults").document(roll_no).get()
                if final_ref.exists:
                    fr = final_ref.to_dict()
                    session_records.append({
                        "date":             sess_data.get("date", ""),
                        "status":           fr.get("status", "absent"),
                        "avgFocusScore":    fr.get("avgFocusScore", 0),
                        "dominantBehavior": fr.get("dominantBehavior", "unknown"),
                        "phoneUsed":        fr.get("phoneUsed", False),
                    })

            if not session_records:
                continue

            # ── Consecutive absences check ──────────────────────────────
            consecutive_absent = 0
            for r in session_records:
                if r["status"] == "absent":
                    consecutive_absent += 1
                else:
                    break

            if consecutive_absent >= 3 and parent_email:
                existing = db.collection("alerts") \
                             .where("rollNo", "==", roll_no) \
                             .where("type",   "==", "absent") \
                             .where("emailSent", "==", True) \
                             .limit(1).stream()
                already_sent = any(True for _ in existing)

                if not already_sent:
                    perf_data = _build_performance_data(session_records)
                    stu_info  = {
                        **stu_data, "rollNo": roll_no,
                        "subject": subject, "room": room, "teacher": teacher
                    }
                    pdf_out = f"/tmp/report_{roll_no}.pdf"
                    generate_pdf(stu_info, perf_data, pdf_out)

                    body = (
                        f"Dear Parent/Guardian,\n\n"
                        f"This is an automated alert from SmartAttend regarding {name} ({roll_no}).\n\n"
                        f"Your child has been absent for {consecutive_absent} consecutive class sessions "
                        f"in {subject} (Teacher: {teacher or 'N/A'}). "
                        f"Please ensure regular attendance to avoid academic penalties.\n\n"
                        f"A detailed performance report is attached.\n\n"
                        f"Regards,\nSmartAttend System"
                    )
                    sent = send_email(
                        parent_email,
                        f"Attendance Alert – {name} ({subject})",
                        body, pdf_out
                    )

                    alert_ref = db.collection("alerts").document()
                    alert_ref.set({
                        "rollNo":    roll_no,
                        "classId":   class_id,
                        "type":      "absent",
                        "detail":    f"{consecutive_absent} consecutive absences in {subject}",
                        "createdAt": datetime.now(timezone.utc),
                        "emailSent": sent
                    })
                    alerts_created.append({
                        "rollNo": roll_no, "name": name,
                        "type": "absent", "emailSent": sent
                    })

            # ── Declining performance check ──────────────────────────────
            if len(session_records) >= 4:
                perf_scores = []
                for r in session_records[:4]:
                    att = 1.0 if r["status"] == "present" else (0.5 if r["status"] == "late" else 0.0)
                    foc = r["avgFocusScore"] / 100.0
                    perf_scores.append((att * 0.5) + (foc * 0.5))

                declining = all(perf_scores[i] < perf_scores[i-1] for i in range(1, len(perf_scores)))
                critical  = declining and perf_scores[-1] > 0 and (perf_scores[0] < perf_scores[-1] * 0.80)

                if declining:
                    alert_type = "critical_declining" if critical else "declining"
                    existing   = db.collection("alerts") \
                                   .where("rollNo", "==", roll_no) \
                                   .where("type",   "==", alert_type) \
                                   .limit(1).stream()
                    if not any(True for _ in existing):
                        alert_ref = db.collection("alerts").document()
                        alert_ref.set({
                            "rollNo":    roll_no,
                            "classId":   class_id,
                            "type":      alert_type,
                            "detail":    f"Performance declining over last 4 sessions in {subject}",
                            "createdAt": datetime.now(timezone.utc),
                            "emailSent": False
                        })
                        alerts_created.append({
                            "rollNo": roll_no, "name": name,
                            "type": alert_type, "emailSent": False
                        })

    return alerts_created


def _build_performance_data(session_records: list) -> dict:
    """Build performance_data dict from session records."""
    total         = len(session_records)
    present_count = sum(1 for r in session_records if r["status"] in ("present", "late"))
    att_rate      = (present_count / total * 100) if total else 0

    focus_vals = [r.get("avgFocusScore", r.get("focusScore", 0)) for r in session_records]
    focus_avg  = sum(focus_vals) / total if total else 0

    beh_counts = {"focused": 0, "distracted": 0, "phone": 0}
    for r in session_records:
        b = r.get("dominantBehavior", "focused")
        if b in beh_counts:
            beh_counts[b] += 1
    beh_total = sum(beh_counts.values()) or 1
    beh_pct   = {k: v / beh_total * 100 for k, v in beh_counts.items()}

    beh_score  = beh_pct.get("focused", 0)
    perf_score = (att_rate * 0.5) + (focus_avg * 0.3) + (beh_score * 0.2)

    trend = "stable"
    if len(session_records) >= 4:
        scores = []
        for r in session_records[:4]:
            a = 1.0 if r["status"] == "present" else (0.5 if r["status"] == "late" else 0.0)
            s = (a * 50) + (r.get("avgFocusScore", r.get("focusScore", 0)) * 0.5)
            scores.append(s)
        if all(scores[i] < scores[i-1] for i in range(1, len(scores))):
            trend = "critical" if scores[0] < scores[-1] * 0.8 else "declining"
        elif all(scores[i] > scores[i-1] for i in range(1, len(scores))):
            trend = "improving"

    return {
        "sessions":           session_records,
        "attendance_rate":    round(att_rate, 2),
        "avg_focus_score":    round(focus_avg, 2),
        "behavior_breakdown": beh_pct,
        "performance_score":  round(perf_score, 2),
        "trend":              trend
    }