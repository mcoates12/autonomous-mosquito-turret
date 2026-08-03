# Project Update — Tracking and Control Overhaul

It has been a while since I seriously worked on this project, but over the last
few days I came back to it and made what is probably the biggest software jump
the turret has had so far.

The goal for this round of work was not to add the mosquito-detection model yet.
I still want to find an existing model trained on mosquitoes if possible instead
of immediately committing to building and labeling an entire dataset myself.
For now, I wanted to get the camera pipeline, red-dot detection, targeting,
servo control, safety behavior, and troubleshooting workflow as polished as I
could. The red laser dot is still acting as a stand-in target so I can work on
the motion system without mixing AI-model problems into controller problems.

At this point, the tracking is honestly close to perfect. There are still a few
small parameters I can min/max, but the turret went from laggy, jittery, and
occasionally locking onto random objects to tracking fast laser movement very
smoothly.

---

## The Original Performance Problem

The biggest problem when I started was rectification. As soon as I loaded the
GUI, the camera pipeline dropped to around **8 FPS**, even though both AR0234
cameras are capable of delivering **1920 × 1200 UYVY at 60 FPS**.

The debug timing made the bottleneck obvious:

- Camera capture: roughly **35–37 ms**
- Full-frame rectification: roughly **80.8 ms**
- Red-dot detection: roughly **18–19 ms**

The software was doing expensive stereo work in the main tracking path whether
it was needed or not. That meant the detector and controller were constantly
working from old frames, which made the entire turret feel delayed.

I reworked the pipeline so that:

- Each camera has a latest-frame capture thread instead of making the tracking
  loop wait on every frame.
- Old frames are dropped instead of building a backlog.
- Red-dot detection runs on the raw tracking-camera image.
- Full stereo rectification and disparity work happen only on scheduled depth
  updates.
- Depth processing runs in a separate worker and consumes only the newest
  request.
- Rectification maps use OpenCV's faster fixed-point representation.
- Preview rendering is demand-driven, so a future headless system does not pay
  for GUI work it is not using.

This removed full-frame rectification from the critical tracking path.

---

## Jetson Camera Acceleration

The next major improvement was the camera conversion pipeline. The Jetson was
spending too much time doing UYVY-to-BGR conversion in software, so I added an
NVIDIA VIC-accelerated GStreamer path using `nvv4l2camerasrc` and `nvvidconv`.
The software still has a portable conversion fallback if the accelerated path
cannot open.

There was an additional caps-negotiation problem that caused GStreamer critical
warnings and fallback behavior. I corrected the pipeline so the caps are fixed
properly before OpenCV receives the frames.

After these changes, the tracking pipeline went from roughly **8 FPS** to around
**55–60 FPS**. The browser preview can intentionally run slower to save network
and encoding overhead, but the actual camera capture, detection, and controller
continue running at full speed on the Jetson.

---

## Camera Exposure and Lighting

One of the early tests looked almost completely black. I originally thought
something was wrong with the camera, but the manual exposure settings were just
far too dark for the room.

The software now starts with:

- Automatic exposure enabled
- Automatic white balance enabled
- Gain set to **1**
- Low-latency camera mode enabled

Manual controls are still available when I need repeatable detector tests. The
starting manual exposure is **100**, which is approximately **10 ms** using the
standard V4L2 units. The GUI caps manual exposure at **160**, or approximately
**16 ms**, so a long exposure cannot silently force a nominal 60 FPS stream to
run slower.

Sunlight was still making the red dot difficult to detect. It turned out that
the laser pointer battery was also weak. Replacing the battery made a huge
difference and gave the detector a much cleaner target.

---

## Headless Browser Dashboard

The Qt GUI is only a troubleshooting tool. The final turret is supposed to run
headless, so I added a browser dashboard that runs directly from the Jetson.

The Jetson now handles:

- Camera capture
- Detection
- Stereo depth work
- Servo control
- Safety logic

My PC only receives a reduced-rate MJPEG troubleshooting preview and sends
validated parameter changes back to the Jetson. I can access it through VS
Code's SSH port forwarding or a normal SSH tunnel, so I no longer need to fight
with VNC, X11 forwarding, black screens, or Qt display errors.

The dashboard also has a heartbeat fail-safe. If the browser connection
disappears for five seconds, tracking is disabled.

No private IP address or network credentials are stored in the repository.

---

## Red-Dot Detector Improvements

The original red detector worked in ideal conditions, but it would either miss
the dot in bright lighting or lock onto things like the brown cardboard box at
the bottom of the camera image.

I added several layers of filtering:

- Brightness threshold
- Saturation threshold
- Minimum and maximum blob area
- Peak-brightness gate
- Local brightness-contrast gate
- Local red-contrast gate
- Hard area gate
- Image-edge exclusion

The detector now preserves tiny laser candidates instead of removing them with
an overly aggressive morphology pass. It also rejects broad reddish or brown
objects that do not have the small, bright, locally red appearance of the laser
dot.

Candidates near the image edge are ignored because partial blobs and bright
objects cut off by the frame were causing false locks.

The current detector values are:

- Brightness threshold: **80**
- Saturation threshold: **35**
- Minimum area: **1 pixel**
- Maximum area: **3000 pixels**
- Peak brightness gate: **140**
- Local contrast gate: **25**
- Local red-contrast gate: **12**
- Image-edge exclusion: **48 pixels**
- Hard area gate: **3000 pixels**

These are troubleshooting values for the red-dot detector, not permanent
mosquito-model parameters.

---

## Target Identity and Reacquisition

Another serious problem was target switching. If the dot moved quickly or
temporarily disappeared, the detector could jump to an unrelated red object and
then keep moving its search anchor toward that false target.

The detector now preserves the last confirmed target identity. During active
tracking, a new candidate must stay within the configured **300-pixel
reacquisition radius** of the last accepted target. Rejected outliers do not get
to move the anchor.

If the dot is lost, the turret holds its position instead of searching the
entire frame and choosing a random red object. Pressing **Stop** and then
**Start** intentionally clears the old identity and allows a fresh full-frame
acquisition.

This logic is detector-neutral, so the same target-observation interface can be
used by a future mosquito model.

---

## Servo Controller Rewrite

I separated servo control from the camera frame loop. The Dynamixel bus now has
its own fixed-rate controller thread that always consumes the newest target
error instead of sending commands whenever a camera frame happens to finish.

The controller now includes:

- Fixed-rate **60 Hz** goal commands
- Latest-target-only behavior
- Stale-target rejection
- Time-scaled movement that does not change with frame rate
- Safe startup without jumping to an old command
- Explicit torque-off behavior
- Dynamixel bus watchdog configuration
- Position, velocity, load, voltage, temperature, and hardware-error telemetry
- Thread-safe shutdown and fault latching

Position and velocity feedback now run at **30 Hz**, while full health telemetry
runs at **1 Hz**. Goal commands remain at 60 Hz. This reduces unnecessary bus
traffic because a servo feedback transaction was taking roughly **12–14 ms**
inside a **16.7 ms** controller period.

The 1 Hz health transaction already includes position and velocity, so the
software reuses that information instead of immediately performing another
feedback read.

---

## Servo Limits and Safety

The software reads the position limits stored in the servos instead of assuming
generic values.

Current limits:

- Pan ID 1: **0–4095 ticks** (approximately 0–359.91°)
- Tilt ID 2 hardware range: **0–2190 ticks** (approximately 0–192.48°)
- Temporary software tilt guard: **708–2190 ticks** (approximately
  62.23–192.48°)

The extra tilt minimum is a conservative mechanical restriction based on the
current assembly. I can change it later after more physical testing. The
controller refuses to arm if the startup feedback is outside the effective safe
range.

Tracking starts disabled, and confirmed controller faults torque the system off
and remain latched until I explicitly re-arm it.

---

## Dynamixel Communication Faults

While testing fast movement, the servo system occasionally shut itself off.
The first problem was an `Incorrect status packet` response during the hardware
status read. I added three bounded attempts with a short delay between them.
A persistent communication problem still shuts down and latches the controller.

Later, the log reported:

```text
pan=0xf3 tilt=0x13
```

Those values looked like hardware faults, but they were impossible XL430 error
bytes. The only valid hardware-error bits combine into mask `0x3D`; bits 1, 6,
and 7 are reserved and must always be zero. Both reported values contained
reserved bits.

The software now validates hardware-error bytes against `0x3D`:

- Impossible reserved-bit values are retried.
- Three persistent invalid values become a latched communication fault.
- Valid nonzero hardware-error bits remain immediately fatal.
- The controller never automatically re-enables torque after a confirmed
  failure.

This keeps the safety behavior while preventing one corrupted telemetry byte
from pretending to be two simultaneous servo failures.

---

## Tracking and Damping

The first tracking controller was essentially proportional-only. It moved
toward the current pixel error, but fast laser movement caused oscillation and
single-pass overshoot.

I first tuned the basic values to:

- Pan gain: **0.003 degrees/pixel**
- Tilt gain: **0.003 degrees/pixel**
- Maximum step: **0.60°** at the 60 Hz reference rate
- Target smoothing: **35 ms**
- Deadband: **25 pixels**
- Profile velocity: **200**
- Profile acceleration: **30**

Reducing the pan gain removed the repeated side-to-side oscillation. Reducing
the smoothing delay and maximum step reduced the remaining overshoot, but a
fast pass could still cross the target once and correct back.

I then added independent filtered derivative damping for both axes:

- Pan damping: **0.006**
- Tilt damping: **0.003**

The damping uses the timestamped rate of change of pixel error. When the error
is closing quickly, it begins braking before the turret crosses the target. It
is filtered to reduce detector noise and resets whenever the target is lost,
becomes stale, or tracking is stopped.

This is not a full traditional PID loop, and I do not currently plan to add an
integral term. The controller already accumulates position commands, and an
additional integral term could create windup. The current proportional plus
derivative behavior is smooth, responsive, and very close to where I want it.

---

## Mechanical Parameters Recorded

I rebuilt the assembly in SolidWorks so I could estimate the moving mass,
center of gravity, and moment of inertia for each axis.

### Tilt axis

- Moving mass: **0.05070 kg**
- Weight: approximately **0.497 N**
- Perpendicular COG radius: **0.03919 m**
- Moment of inertia: **9.0293e-5 kg·m²**
- Maximum estimated gravity torque: approximately **0.01948 N·m**

This model includes the two camera/lens assemblies and moving tilt bracket. It
does not yet include all fasteners, wiring, ribbon cables, laser hardware, or
omitted rotating horn hardware.

### Pan axis

- Moving mass: **0.11850 kg**
- Radial COG offset in the upright pose: approximately **6.05 mm**
- Upright moment of inertia: **6.54e-5 kg·m²**
- Estimated worst-case inertia over the tilt range: **1.69e-4 kg·m²**
- Conservative controller-design value: **1.7e-4 kg·m²**

These values give me a much better starting point for future controller sizing
and acceleration limits than guessing from the servo specifications.

---

## Testing and Code Quality

The tracking code is now split into reusable pieces instead of being one giant
GUI-dependent loop. Camera pipelines, target observations, filtering, depth,
servo control, dashboard validation, and motion math can be tested separately.

The repository currently has **47 passing automated tests** covering areas such
as:

- Camera pipeline selection and fallback
- Frame-rate-independent filtering and motion
- Position conversion and safe limits
- Dashboard input validation and heartbeat behavior
- Stale-target handling
- Target identity and reacquisition radius
- Proportional/derivative damping math
- Servo torque-off and fault latching
- Communication retries
- Hardware-status mask validation
- Decoupled command, feedback, and health rates

This is especially important because the final detector may change from a laser
heuristic to YOLO or another model, but the controller and safety behavior
should not have to be rewritten.

---

## Current Status and Next Steps

The turret currently tracks the laser dot at approximately **55–60 FPS** and
moves smoothly without the repeated oscillation I was seeing before. Fast
movement can still use some very small tuning changes, but I would describe the
current behavior as near perfect for this stage of the project.

My next steps are:

1. Make small pan and tilt damping adjustments and record the final values.
2. Record controlled stationary, slow-sweep, fast-sweep, reversal, and
   irregular-motion tests.
3. Measure overshoot, settling time, end-to-end latency, backlash, and any axis
   coupling instead of judging everything only by eye.
4. Establish the physical aiming point in camera coordinates rather than
   permanently assuming the numerical center of the image.
5. Find and integrate a mosquito-detection model behind the existing
   detector-neutral target interface.
6. Add velocity prediction/feedforward after the measured controller baseline
   is complete. A Kalman filter may eventually provide the state estimate, but
   it is not the feedforward loop by itself.
7. Keep the troubleshooting dashboard for development while moving the final
   system toward fully headless operation.

This round of work was a lot of debugging, but the result finally feels like a
real tracking platform instead of a camera script connected to two servos. The
AI model is still ahead, but the foundation it will plug into is dramatically
better than it was before.
