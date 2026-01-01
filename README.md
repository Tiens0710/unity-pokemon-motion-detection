# 🎮 Pokémon Motion Detection Game (Unity)

> A Pokémon-inspired game built with **Unity**, controlled by **real-time motion detection** using a camera.

---

## 📌 Overview

This project explores **motion-based gameplay** by allowing players to control a Pokémon-style character using **body movements and gestures** instead of traditional keyboard or controller inputs.

The system combines:

* **Unity** for gameplay and visuals
* **Python + OpenCV** for motion detection
* **Webcam input** to track player movement

The goal is to create a more **immersive and physical interaction** with the game world.

---

## 🎥 Gameplay Demo

> ⚠️ **GitHub does not reliably support embedded `<video>` tags in README files.**

### ▶️ Watch the demo video

Click the link below to watch or download the gameplay demo:

👉 **[Watch DemoGame.mp4](./DemoGame.mp4)**

Or view via raw link:

```
https://github.com/Tiens0710/REPO_NAME/raw/master/DemoGame.mp4
```

> If the video does not play in the browser, GitHub will automatically download it.

---

## 🧠 How Motion Detection Works

1. The camera captures live video of the player
2. `motiondetection.py` processes frames using **OpenCV**
3. Player movements are analyzed to determine:

   * Direction (left / right / forward / backward)
   * Actions (gesture-based interaction)
4. Signals are sent to Unity to update the character state in real time

This allows **hands-free gameplay** using natural body movement.

---

## 🕹️ Features

* 🎥 Real-time motion detection via webcam
* 🧍 Character movement using body gestures
* 🎮 Unity-based Pokémon-style gameplay
* 🧠 Computer Vision integration
* 🔌 Extensible architecture for AR / AI upgrades

---

## 📂 Project Structure

```text
.
├── DemoGame.mp4        # Gameplay demo video
├── motiondetection.py # Python script for motion detection (OpenCV)
└── README.md           # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Requirements

* Python 3.8+
* Unity Hub + compatible Unity version
* Webcam

Install Python dependencies:

```bash
pip install opencv-python numpy
```

### 2️⃣ Run Motion Detection

```bash
python motiondetection.py
```

### 3️⃣ Run Unity Game

* Open the Unity project
* Start the main scene
* Ensure the webcam is active and the Python script is running

---

## 🎯 Project Goals

* Apply **Computer Vision** to interactive games
* Explore **motion-based control systems**
* Build a foundation for:

  * AR / VR games
  * Pokémon GO–style experiences
  * Educational & rehabilitation games

---

## 🔮 Future Improvements

* Full body pose detection (MediaPipe / PoseNet)
* Gesture-based Pokémon battles
* Multiplayer motion tracking
* AR mode with real-world interaction

---

## 👨‍💻 Author

**Tiến Nguyễn**
GitHub: [https://github.com/Tiens0710](https://github.com/Tiens0710)

---

⭐ If you find this project interesting, feel free to star the repository or use it as a reference for motion-based game development.
