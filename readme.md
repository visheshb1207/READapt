<h1 align = 'center'>READapt</h1>

# AI-Powered Adaptive Reading Assistant for Dyslexic Learners

READapt is an AI-driven assistive reading system designed to support dyslexic learners through **pronunciation analysis**, **attention tracking**, and **dynamic, multisensory feedback**. It addresses the limitations of traditional one-size-fits-all reading tools by adapting to each learner’s performance, engagement, and focus in real time.

---

## 🎯 Objectives

- Detect pronunciation accuracy using modern ASR (e.g., Whisper, Google Speech API)
- Analyze learner attention using webcam-based gaze and focus tracking
- Adapt reading difficulty using machine learning based on performance
- Provide multisensory support (text-to-speech, color cues, instant hints)
- Deliver warm, human-like motivational feedback through an AI persona

---

## 🧩 Key Features

### 🔊 Pronunciation Analysis
- Speech-to-text transcription
- Detection of mispronunciations, skipped words, and fluency issues
- Instant corrective feedback

### 👁 Attention & Engagement Tracking
- Webcam-based gaze and focus detection
- Identifies distraction, fatigue, and attention loss

### 🔁 Adaptive Learning Engine
- Automatically adjusts:
  - Word and sentence difficulty
  - Reading speed
  - Font size, spacing, and visual aids
- Learner-specific personalization

### 🌈 Multisensory Assistance
- Text-to-speech (TTS)
- Color-coded cues and syllable highlighting
- Real-time hints and guidance

### 🤖 AI Mentor Persona
- Personalized encouragement
- Gentle error correction
- Session summaries and progress insights

---

## 🛠 Tech Stack

- **Frontend / UI:** Streamlit  
- **ASR:** Whisper / Google Speech API  
- **Computer Vision:** OpenCV, MediaPipe  
- **ML & Analytics:** Python, scikit-learn  
- **Language:** Python 3.11+  

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/visheshb1207/READapt.git
cd READapt
```

### 2️⃣ Create & Activate Virtual Environment 

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Ensure you are in the project root directory.

```bash
streamlit run app.py
```

(If your main file has a different name, replace `app.py` accordingly.)

Open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

## 📤 GitHub Push Instructions

### 1️⃣ Initialize Git (if not already)

```bash
git init
```

### 2️⃣ Add Files

```bash
git add .
```

### 3️⃣ Commit Changes

```bash
git commit -m "Initial commit: READapt AI-based dyslexia reading assistant"
```

### 4️⃣ Add Remote Repository

```bash
git remote add origin https://github.com/visheshb1207/READapt.git
```

### 5️⃣ Push to GitHub

```bash
git push -u origin main
```

(Use `master` instead of `main` if applicable.)

---

## 📌 Future Enhancements

- Advanced fluency scoring models
- Classroom & educator dashboards
- Multilingual reading support
- Long-term learner performance analytics

---

## 📄 License

This project is licensed under the MIT License (or your preferred license).
