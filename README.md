# 👋 SignVision.AI  
*Real-time sign language interpreter using hand gestures + speech output*

![Python](https://img.shields.io/badge/Python-3.8-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-CV-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-UI-informational)
![gTTS](https://img.shields.io/badge/Text2Speech-Google-green)
![Status](https://img.shields.io/badge/Live-Demo%20Ready-brightgreen)

---

## 🚀 About
**SignVision.AI** is an AI-powered real-time sign language interpreter.  
It captures hand gestures via webcam, identifies ASL signs, and converts them into **spoken output** using Google TTS — making communication more accessible for the hearing/speech impaired.

---

## 🧠 Tech Stack

- 🖐️ [MediaPipe](https://google.github.io/mediapipe/) for hand landmark detection  
- 🧠 Custom-trained model for ASL letter & word classification  
- 🎥 [OpenCV](https://opencv.org/) for UI + gesture visualization  
- 🔊 [gTTS](https://pypi.org/project/gTTS/) for speech output  
- 🚀 Deployable on AWS Lambda for scaling

---

## ✨ Features

- 🔍 Detects static hand gestures for ASL letters  
- 🧠 Achieves **92% accuracy (letter)** and **85% (word-level)**  
- ⚡ Real-time recognition (<100ms latency)  
- 🔊 Speaks recognized signs out loud  
- 🔁 Supports dataset expansion + retraining

---

## 📦 Installation

```bash
git clone https://github.com/Sujal-py3/SignVision.AI.git
cd SignVision.AI
pip install -r requirements.txt
python app.py

##Demo
![SignVision Demo](signvision_demo.gif)


