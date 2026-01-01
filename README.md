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

Below is a short demo showing character movement and interaction controlled by body motion:

<video src="./DemoGame.mp4" controls width="720"></video>

> If the video does not autoplay on your device, download `DemoGame.mp4` directly from the repository.

---

## 🧠 How Motion Detection Works

1. The camera captures live video of the player
2. `motiondetection.py` processes frames using **OpenCV**
3. Player movements are analyzed to determine:

   * Direction (left / right / forward / backward)
   * Actions (gesture-based interaction)
4. Signals are sent to Unity to update the character state in real time

This allows **hands-free gameplay** using natural body movement.

## 👨‍💻 Author

**Tiến Nguyễn**
GitHub: [https://github.com/Tiens0710](https://github.com/Tiens0710)

---

⭐ If you find this project interesting, feel free to star the repository or use it as a reference for motion-based game development.
