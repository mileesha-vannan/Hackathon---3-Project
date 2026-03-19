# 🎵 AI Music Grading System v2.0

A self-improving machine learning system that analyzes audio files and assigns quality scores based on extracted features and user feedback.

---

## 🚀 Overview

This project is an **AI-powered music evaluation system** that:

- Extracts audio features from uploaded tracks
- Allows human grading
- Learns from feedback using ML models
- Improves predictions over time
- Displays results via an interactive dashboard

---

## 🧠 How It Works

1. Upload Audio
2. Feature Extraction
3. Human Grading
4. Model Training
5. Prediction
6. Self-Improvement Loop

---

## 📁 Project Structure

├── main.py  
├── config.json  
├── dashboard.html  
├── requirements.txt  
├── uploads/  
├── grading_dataset.csv  
├── best_model.pkl  
├── session_history.json  

---

## ⚙️ Installation

```bash
python3 -m venv music_env
source music_env/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python main.py
```

Open dashboard.html in your browser.

---

## 🌐 API

- GET /api/status
- GET /api/history
- GET /api/models
- POST /api/grade
- POST /api/retrain
- GET /api/config

---

## 🛠️ Tech Stack

- Flask
- scikit-learn
- librosa
- Chart.js

---

## 📌 Notes

- Uses first 30 seconds of audio
- Model improves with more data
