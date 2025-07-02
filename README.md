# Autonomous Mosquito-Killing Turret

A real-time, dual-camera turret system designed to detect, track, and eliminate mosquitoes using AI-based vision, stereo depth estimation, and precision servo control. Built solo from scratch as a self-funded engineering challenge and proof of capability.

---

## Objective

Create a fully autonomous turret capable of:
- Detecting flying mosquitoes in real-time up to 2 meters away
- Tracking targets using stereo vision and depth estimation
- Driving high-speed Dynamixel servos to follow targets
- Executing a "kill" action (via zap or simulated event)
- Implements real-time mosquito path prediction using a Kalman filter to estimate future positions and improve turret accuracy.

---

## System Overview

|    Component     |                   Description                         |
|------------------|-------------------------------------------------------|
| Jetson Orin Nano | Real-time inference, image processing, control logic  |
| 2x AR0234 Cameras| Global shutter cameras for synchronized stereo input  |
| 16mm Lenses      | Narrow FOV for long-range detection accuracy          |
| Dynamixel XL430  | Precision pan/tilt control with speed + torque        |
| YOLOv8n          | Object detection model trained on mosquito imagery    |
| 3D Printed Parts | Custom mounts, housing, and brackets                  |

---

## Features (WIP)

- [x] Hardware purchased
- [ ] Prototype brackets and mounts 3d printed
- [ ] Dual-camera vision setup
- [ ] Servo communication + movement test
- [ ] Depth estimation from stereo input
- [ ] Path prediction via kalman filter
- [ ] Real-time YOLOv8 inference
- [ ] Final turret housing designed
- [ ] Turret Housing 3d printed
- [ ] Multi-step kill logic (detection + motion confirm + depth range)
- [ ] Fail-safe & safety protocols
- [ ] Audible status beeps for states (idle, tracking, kill-ready)

---

## Project Structure

---

> 🔍 [Read the full project backstory → logs/project-intro.md](logs/project-intro.md)

## 👤 Author

**Myles**  
Junior Mechanical Engineering Student  
Self-taught builder | Defense-focused 

Feel free to fork, contribute, or reach out for questions.
