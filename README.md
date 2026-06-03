# AutoVision AI: Autonomous Driving System

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![CARLA](https://img.shields.io/badge/CARLA-0.9.13-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8.0-orange.svg)

An advanced, end-to-end autonomous driving simulation built on the **CARLA Simulator**. This project integrates Computer Vision, Sensor Fusion, and Reinforcement Learning to create a vehicle capable of navigating complex urban environments.

---

## 🌟 Key Features

### 🧠 Decision Making & Control
- **Deep Q-Network (DQN) Agent**: A PyTorch-based RL agent that learns optimal driving policies (throttle, steer, brake) based on environmental states.
- **Behavior-based Navigation**: Integrated CARLA `BehaviorAgent` for high-level path planning and waypoint following.
- **Autonomous Parking**: A dedicated state-machine module for precise parallel and reverse parking.

### 👁️ Perception & Vision
- **YOLOv8 + DeepSORT**: Real-time detection and multi-object tracking of vehicles, pedestrians, and traffic signs.
- **Lane Detection**: Robust HLS color space transformation and Hough Line Transform for lane discipline.
- **Dynamic Speed Control**: Intelligent parsing of speed limit signs to automatically regulate vehicle velocity.

### 🛰️ Sensor Fusion & Prediction
- **LiDAR-Camera Fusion**: Projecting 3D LiDAR point clouds onto 2D camera frames for accurate distance estimation and TTC (Time-to-Collision) calculation.
- **AI Traffic Predictor**: A predictive module that classifies the behavior of nearby vehicles (Braking, Accelerating, Lane Changing) using velocity history.

### 📊 Real-time Monitoring
- **Live Telemetry Dashboard**: A Flask-based web interface showing real-time speed, steering, AI "thoughts", and traffic predictions.

---

## 🛠️ Tech Stack
- **Simulation**: CARLA Simulator
- **Deep Learning**: PyTorch, YOLOv8, DeepSORT
- **Computer Vision**: OpenCV, NumPy
- **Web Backend**: Flask
- **Tracking**: Kalman Filters (via DeepSORT)

---

## 🚀 Installation & Usage

### 1. Prerequisites
- Windows 10/11
- CARLA Simulator (0.9.13 recommended)
- Python 3.7 or 3.8

### 2. Setup Environment
```bash
pip install -r requirements.txt
# Ensure torch, torchvision, ultralytics, opencv-python, and flask are installed.
```

### 3. Running the System
1. **Start CARLA**: Run your CARLA executable.
2. **Start Dashboard**:
   ```bash
   python dashboard_server.py
   ```
3. **Launch AI System**:
   ```bash
   python advanced_drive.py
   ```
4. **View Dashboard**: Open `http://localhost:5000` in your browser.

---

## 🏗️ System Architecture
1. **Sensor Layer**: Camera, LiDAR, and Collision sensors collect raw data.
2. **Perception Layer**: YOLO detects objects; Lane Detector finds road boundaries.
3. **Fusion & Prediction Layer**: LiDAR data is fused with camera bboxes; Traffic Predictor forecasts neighbor movements.
4. **Planning Layer**: DQN + BehaviorAgent decide the best action.
5. **Control Layer**: Commands (Throttle/Steer/Brake) are sent back to the CARLA vehicle.

---

## 🏁 Conclusion
This project demonstrates a production-grade approach to self-driving technology, combining classical computer vision with modern deep learning and reinforcement learning techniques. Created for Hackathon 2026.
