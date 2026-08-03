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
| YOLO             | Object detection model trained on mosquito imagery    |
| 3D Printed Parts | Custom mounts, housing, and brackets                  |

---

## Features (WIP)

- [x] Hardware purchased
- [x] Prototype brackets and mounts 3d printed
- [x] Dual-camera vision setup
- [x] Servo communication + movement test
- [x] Depth estimation from stereo input
- [ ] Path prediction via kalman filter
- [ ] Real-time AI inference
- [ ] Final turret housing designed
- [ ] Turret Housing 3d printed
- [ ] Multi-step kill logic (detection + motion confirm + depth range)
- [ ] Fail-safe & safety protocols
- [ ] Audible status beeps for states (idle, tracking, kill-ready)

---

## Project Structure

---

## Browser troubleshooting dashboard

The recommended troubleshooting interface runs on the Jetson without a Linux
desktop or VNC session. Camera capture, detection, and servo control stay on the
Jetson; the PC receives a reduced-rate diagnostic MJPEG preview and sends only
validated parameter changes.

On the Jetson:

```bash
cd ~/Documents/autonomous-mosquito-turret
git pull --ff-only origin main
python3 src/laser_follow_web.py
```

Tracking starts disabled. Keep the Jetson terminal open, and press `Ctrl+C` for
a clean shutdown and torque-off.

When using VS Code Remote SSH, open the **Ports** panel, choose **Forward a
Port**, enter `8080`, then open the forwarded address in the PC browser.

Alternatively, keep this command running in a Windows PowerShell window:

```powershell
ssh -N -L 8080:127.0.0.1:8080 myles@192.168.1.141
```

PowerShell showing no prompt or output means the tunnel is working. Open
<http://127.0.0.1:8080> in the PC browser. The dashboard automatically disables
tracking if its heartbeat disappears for five seconds.

For a trusted private LAN only, the dashboard can be exposed directly with
`python3 src/laser_follow_web.py --bind 0.0.0.0`, then opened at
`http://JETSON_IP:8080`. This mode has no authentication, so the SSH/VS Code
forward is preferred.

The diagnostic stream defaults to 10 FPS at 960 pixels wide and does not set
the camera capture rate. On a heavily loaded Jetson, reduce dashboard overhead
with `--preview-fps 5 --preview-width 640`.

During active tracking, reacquisition remains anchored to the last accepted
target. If the dot is lost, the turret holds instead of switching to a distant
red object. Press **Stop**, place the dot at the new location, and press
**Start** to deliberately clear that identity and perform a fresh full-frame
acquisition. The dashboard's `Reacquire radius` controls the allowed distance.

The outer motion loop uses independent pan and tilt proportional gains plus
filtered derivative damping. Damping is calculated from timestamped target
error, so it does not depend on camera or controller frame rate. It resets on
target loss, stale data, or tracking stop. The starting values are pan/tilt
gain `0.003`, pan damping `0.006`, tilt damping `0.003`, smoothing `35 ms`, and
maximum step `0.60 deg`. Raise damping gradually to reduce single-pass
overshoot; excessive damping can produce twitch or hesitation.

The runtime prefers the Jetson `nvv4l2camerasrc`/`nvvidconv` VIC conversion
path and automatically falls back to the portable software pipeline if it is
unavailable. In the preview overlay, `FPS` is the processed tracking rate and
`src` is the selected camera's independently measured delivery rate.

> 🔍 [Read the full project backstory → logs/project-intro.md](logs/project-intro.md)

## 👤 Author

**Myles**  
Junior Mechanical Engineering Student  
Self-taught builder | Defense-focused 

Feel free to fork, contribute, or reach out for questions.
