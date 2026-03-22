# app.py (corrected)
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, session, flash
import os
import cv2
import time
import numpy as np
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from fer import FER
import threading

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Folders
SKETCH_FOLDER = "static/sketches"
EMOTION_FOLDER = "static/emotions"
OBJECT_FOLDER = "static/objects"
for folder in [SKETCH_FOLDER, EMOTION_FOLDER, OBJECT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Models (load once)
print("🔍 Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")
print("😊 Loading FER emotion detector...")
emotion_detector = FER(mtcnn=True)

# DB init
def init_db():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ----------------------- Camera helpers -----------------------
camera_lock = threading.Lock()
camera = None

def get_camera(device_index=0):
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(device_index)
            # small timeout check
            if not camera.isOpened():
                camera.release()
                camera = None
                return None
        return camera

def release_camera():
    global camera
    with camera_lock:
        if camera:
            try:
                camera.release()
            except Exception:
                pass
            camera = None

# ----------------------- ROUTES -----------------------

# Home as first page (shows Guest when not logged in)
@app.route("/")
def index():
    username = session.get("user", "Guest")
    return render_template("home.html", username=username.capitalize())

@app.route("/home")
def home():
    # keep consistent with index
    username = session.get("user", "Guest")
    return render_template("home.html", username=username.capitalize())

# ---------- REGISTER ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip().lower()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        confirm = request.form.get("confirm_password", "")
        if password != confirm:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("register"))

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                        (username, email, password))
            conn.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
        finally:
            conn.close()
    return render_template("register.html")

# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip().lower()
        password = request.form["password"]
        conn = sqlite3.connect("database.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cur.fetchone()
        conn.close()
        if user:
            session["user"] = username
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password!", "danger")
    return render_template("login.html")

# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))

# ---------- SKETCH ----------
@app.route("/sketch", methods=["GET", "POST"])
def sketch():
    """
    Two behaviors:
    1) POST with file -> generate sketch, store as <username>_sketch_... and render sketch_result.html
    2) POST with 'filename' field -> user clicked Save in result page -> ensure file belongs to user & redirect to profile
    """
    if request.method == "POST":
        # Save-from-result action (Save to profile)
        if "filename" in request.form:
            filename = secure_filename(request.form["filename"])
            username = session.get("user", "guest")
            current_path = os.path.join(SKETCH_FOLDER, filename)
            # If file exists and does not start with username_, rename to username_... (safe save)
            if os.path.exists(current_path):
                if not filename.startswith(f"{username}_"):
                    new_name = f"{username}_{filename}"
                    new_path = os.path.join(SKETCH_FOLDER, new_name)
                    try:
                        os.rename(current_path, new_path)
                        flash("Sketch saved to profile!", "success")
                    except Exception:
                        flash("Could not save sketch, try again.", "danger")
                else:
                    flash("Sketch already saved to profile.", "info")
            else:
                flash("Sketch file not found.", "danger")
            return redirect(url_for("profile"))

        # Normal upload -> create sketch
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please upload an image.", "danger")
            return redirect(url_for("sketch"))

        # read image bytes and convert
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                flash("Uploaded file is not a valid image.", "danger")
                return redirect(url_for("sketch"))
        except Exception as e:
            flash("Failed to read the uploaded image.", "danger")
            return redirect(url_for("sketch"))

        # Fast sketch conversion (optimized)
        try:
            max_dim = 800
            h, w = img.shape[:2]
            scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inverted = cv2.bitwise_not(gray)
            blur = cv2.GaussianBlur(inverted, (15, 15), 0)
            sketch_img = cv2.divide(gray, 255 - blur, scale=256)
        except Exception as e:
            flash(f"Error processing image: {e}", "danger")
            return redirect(url_for("sketch"))

        username = session.get("user", "guest")
        sketch_name = f"{username}_sketch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        sketch_path = os.path.join(SKETCH_FOLDER, sketch_name)
        cv2.imwrite(sketch_path, sketch_img)

        # store last_sketch in session for quick save
        session["last_sketch"] = sketch_name
        return render_template("sketch_result.html", sketch_image=sketch_name)

    # GET
    return render_template("sketch.html")

@app.route("/save_sketch", methods=["POST"])
def save_sketch():
    # optional API-style save (AJAX) - keeps backward compatibility
    filename = session.get("last_sketch")
    if not filename:
        return jsonify({"message": "❌ No sketch to save."}), 400
    username = session.get("user", "guest")
    path = os.path.join(SKETCH_FOLDER, filename)
    if os.path.exists(path):
        # ensure file has username prefix
        if not filename.startswith(f"{username}_"):
            new_name = f"{username}_{filename}"
            os.rename(path, os.path.join(SKETCH_FOLDER, new_name))
            session["last_sketch"] = new_name
        return jsonify({"message": "✅ Sketch saved to your profile!"})
    return jsonify({"message": "❌ File not found."}), 404

# ---------- EMOTION ----------
def gen_frames_emotion():
    cap = get_camera()
    if cap is None:
        # yield a single placeholder image or stop generator
        blank = 255 * np.ones((240, 320, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", blank)
        frame = buffer.tobytes()
        while True:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = emotion_detector.detect_emotions(rgb)
                for det in detections:
                    box = det.get("box")
                    emotions = det.get("emotions", {})
                    if box and emotions:
                        x, y, w, h = box
                        top_em = max(emotions, key=emotions.get)
                        conf = int(emotions[top_em] * 100)
                        label = f"{top_em} ({conf}%)"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            except Exception:
                # ignore single-frame errors from FER
                pass
            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        release_camera()

@app.route("/emotion")
def emotion_page():
    username = session.get("user", "Guest")
    return render_template("emotion.html", username=username.capitalize())

@app.route("/video_feed_emotion")
def video_feed_emotion():
    return Response(gen_frames_emotion(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/capture_emotion", methods=["POST"])
def capture_emotion():
    cap = get_camera()
    if cap is None:
        return jsonify({"error": "Camera not available"}), 500
    success, frame = cap.read()
    if not success:
        return jsonify({"error": "Failed to capture image."}), 500
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = emotion_detector.detect_emotions(rgb)
        if detections:
            emotions = detections[0].get("emotions", {})
            detected_emotion = max(emotions, key=emotions.get) if emotions else "neutral"
        else:
            detected_emotion = "neutral"
    except Exception:
        detected_emotion = "neutral"

    username = session.get("user", "guest")
    filename = f"{username}_{detected_emotion}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(EMOTION_FOLDER, filename)
    cv2.imwrite(filepath, frame)
    session["last_emotion"] = filename
    return jsonify({"emotion": detected_emotion})
@app.route("/upload_emotion", methods=["POST"])
def upload_emotion():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections = emotion_detector.detect_emotions(rgb)
    if detections:
        emotions = detections[0]["emotions"]
        detected_emotion = max(emotions, key=emotions.get)
    else:
        detected_emotion = None

    username = session.get("user", "guest")
    filename = f"{username}_{detected_emotion or 'neutral'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(EMOTION_FOLDER, filename)
    cv2.imwrite(filepath, img)

    return jsonify({"emotion": detected_emotion})

@app.route("/save_emotion", methods=["POST"])
def save_emotion():
    filename = session.get("last_emotion")
    if not filename:
        return jsonify({"message": "❌ No emotion to save."}), 400
    path = os.path.join(EMOTION_FOLDER, filename)
    if os.path.exists(path):
        return jsonify({"message": "✅ Emotion saved to profile!"})
    return jsonify({"message": "❌ Emotion file not found."}), 404

# ---------- OBJECT ----------
def gen_frames_object():
    cap = get_camera()
    if cap is None:
        blank = 255 * np.ones((240, 320, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", blank)
        frame = buffer.tobytes()
        while True:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue
            try:
                results = yolo_model(frame)
                if len(results) > 0:
                    boxes = results[0].boxes
                    for box in boxes:
                        try:
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = xyxy
                            cls_id = int(box.cls[0].cpu().numpy())
                            label = yolo_model.names.get(cls_id, str(cls_id))
                            conf = float(box.conf[0].cpu().numpy())
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        except Exception:
                            continue
            except Exception:
                pass
            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        release_camera()

@app.route("/object")
def object_page():
    username = session.get("user", "Guest")
    return render_template("object.html", username=username.capitalize())

@app.route("/video_feed_object")
def video_feed_object():
    return Response(gen_frames_object(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/save_object", methods=["POST"])
def save_object():
    username = session.get("user", "guest")
    filename = f"{username}_object_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    open(os.path.join(OBJECT_FOLDER, filename), "a").close()
    return jsonify({"message": "✅ Object saved to profile!"})

# ---------- PROFILE ----------
@app.route("/profile")
def profile():
    username = session.get("user", "guest")
    # list files that begin with username_ (if any)
    sketches = [f for f in os.listdir(SKETCH_FOLDER) if f.startswith(username)]
    emotions = [f for f in os.listdir(EMOTION_FOLDER) if f.startswith(username)]
    objects = [f for f in os.listdir(OBJECT_FOLDER) if f.startswith(username)]
    return render_template("profile.html", username=username, sketches=sketches, emotions=emotions, objects=objects)

# Delete file endpoint
@app.route("/delete_file/<category>/<filename>", methods=["POST"])
def delete_file(category, filename):
    folder_map = {
        "sketch": SKETCH_FOLDER,
        "emotion": EMOTION_FOLDER,
        "object": OBJECT_FOLDER
    }
    folder = folder_map.get(category)
    if not folder:
        return jsonify({"message": "❌ Invalid category."}), 400
    safe_name = secure_filename(filename)
    path = os.path.join(folder, safe_name)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"message": f"✅ {category.capitalize()} deleted successfully!"})
    else:
        return jsonify({"message": "⚠️ File not found."}), 404

# ---------- Errors ----------
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("500.html"), 500

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
