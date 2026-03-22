🎯 Sketch–Emotion–Object Analyzer

📌 Overview

The Sketch–Emotion–Object Analyzer is an AI-powered web application that combines:

🎨 Image to Sketch Conversion

😊 Real-Time Emotion Detection

🎯 Object Detection


All functionalities are integrated into a single platform using Flask, making it interactive and user-friendly.


---

🚀 Features

✨ Convert images into realistic pencil sketches
😊 Detect human emotions in real-time using webcam
🎯 Identify objects with high accuracy using YOLOv8
👤 User authentication system (Login/Register/Profile)
💾 Save and manage outputs in user profile


---

🛠️ Tech Stack

💻 Frontend

HTML5

CSS3

JavaScript


⚙️ Backend

Python (Flask)

OpenCV

FER (Facial Emotion Recognition)

YOLOv8 (Ultralytics)


🗄️ Database

SQLite3



---

📂 Project Structure

SKETCH-EMOTION-OBJECT-ANALYZER/
│
├── static/
│   ├── backgrounds/
│   ├── css/
│   │   └── styles.css
│   ├── emotions/
│   ├── images/
│   ├── js/
│   ├── models/
│   ├── objects/
│   ├── profile/
│   ├── profile_saves/
│   ├── profile_uploads/
│   ├── profiles/
│   ├── sketches/
│
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── profile.html
│   ├── sketch.html
│   ├── sketch_result.html
│   ├── emotion.html
│   ├── object.html
│
├── app.py
├── requirements.txt
├── yolov8n.pt


---

⚙️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/your-username/sketch-emotion-object-analyzer.git

cd sketch-emotion-object-analyzer


---

2️⃣ Install Dependencies

pip install -r requirements.txt


---

3️⃣ Run the Application

python app.py


---

4️⃣ Open in Browser

http://127.0.0.1:5000/


---

🧠 Modules

🎨 Sketch Module

Converts uploaded images into pencil sketches using OpenCV techniques.

😊 Emotion Detection Module

Detects facial expressions like happy, sad, angry using FER.

🎯 Object Detection Module

Uses YOLOv8 to detect and label objects in real-time.

👤 User Module

Handles user registration, login, and profile management.


---

🗄️ Database

SQLite3 database is used to store:

User details

Activity logs

Emotion detection results

Object detection records



---

🔮 Future Enhancements

📱 Mobile application (Android/iOS)

☁️ Cloud deployment

🎤 Voice-based emotion detection

📊 Analytics dashboard



---

🎯 Conclusion

This project demonstrates the integration of Artificial Intelligence and Computer Vision into a real-time web application. It provides accurate, fast, and user-friendly results.


---
