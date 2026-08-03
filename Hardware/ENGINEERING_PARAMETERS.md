# Turret Engineering Parameters

Last updated: 2026-08-02

This file records measured and modeled parameters used for controller sizing,
tracking development, and future calibration. Values marked preliminary must be
remeasured after the final lenses, wiring, fasteners, and accessory hardware are
installed.

## Camera system

- Camera count: 2
- Sensor: onsemi AR0234 global shutter
- Capture format: 1920 x 1200, UYVY
- Confirmed capture rate: 60 FPS
- Pixel size: 3 micrometers
- Lens holder: expected M12 x 0.5 S-mount; verify against the exact camera module
  before ordering replacements
- Current focus: manually adjusted by threading each lens in or out
- Final replacement lens focal length: not yet selected
- The README's 16 mm lens entry describes an intended configuration, not a
  verified final optical configuration

The stored camera calibration has approximately 1,105 pixel focal length. With
3 micrometer pixels, this implies that the lenses used for those images were
approximately 3.3 mm focal length. This estimate is approximate because focus,
distortion, and calibration quality affect the fitted intrinsics.

### Known V4L2 controls

- `/dev/video1` identifies Camera 1 (`ar0234 10-0042`) and reports
  1920 x 1200 UYVY capture at 60 FPS
- Exposure modes: full-frame automatic, manual, and ROI automatic
- Absolute exposure range: 1 to 10,000
- Gain range: 1 to 40
- Automatic and fixed white balance are supported
- Low-latency mode is supported
- Current software startup: automatic exposure and automatic white balance,
  gain 1, and low-latency mode enabled
- Manual tuning starts from exposure 100 (10 ms under standard V4L2 units) and
  fixed white balance 4600 when the operator enables those modes
- The GUI caps manual exposure at 160 (16 ms) to avoid silently reducing a
  nominal 60 FPS stream through excessive exposure time

## Stereo calibration

Historical calibration record:

- Captured stereo pairs: 22
- Usable stereo pairs: 21
- Checkerboard: 9 x 6 inner corners
- Measured square size: 18 mm
- Left reprojection RMS: 0.1703 pixels
- Right reprojection RMS: 0.1772 pixels
- Stereo reprojection RMS: 0.5314 pixels
- Reported corrected baseline: 0.0500 m

Important active-file mismatch:

- `src/stereo_calib/stereo_calibration_full.npz` currently embeds a 0.040 m
  square size and a 0.111211 m baseline.
- Scaling that baseline by 18/40 gives 0.050045 m, which agrees with the
  historical corrected result.
- The active file therefore has the old scale and must not be trusted for
  metric depth. It would report distances approximately 2.22 times too large.
- Exposure and brightness changes do not inherently invalidate geometric
  calibration, but lens focus changes, lens replacement, or camera movement do.
- Perform the final stereo calibration only after installing, focusing, and
  mechanically locking the final matched lenses.

## Servos and mechanical limits

- Servo model: Dynamixel XL430-W250
- Protocol: 2.0
- Operating mode: position control
- Serial device: `/dev/ttyUSB0`
- Baud rate: 1,000,000
- Pan servo ID: 1
- Tilt servo ID: 2
- Pan hardware range: 0 to 4095 ticks (approximately 0 to 359.91 degrees)
- Tilt hardware range: 0 to 2190 ticks (approximately 0 to 192.48 degrees)
- Temporary software tilt guard: 708 to 2190 ticks (approximately 62.23 to
  192.48 degrees)
- The controller refuses to arm if startup feedback is outside its effective
  software limits.

Known Dynamixel gain values from Wizard screenshots, for both servos:

- Velocity P Gain: 100
- Position I Gain: 0
- Feedforward 2nd Gain: 0

Still to record:

- Velocity I Gain
- Position D Gain
- Position P Gain
- Feedforward 1st Gain

## Preliminary tilt-axis properties

The modeled tilt payload includes both modeled camera/lens assemblies and the
moving tilt bracket.

- Moving mass: 0.05070 kg
- Weight under standard gravity: approximately 0.497 N
- COG relative to `CS_Tilt`:
  - X: -0.01722 m
  - Y: -0.00020 m
  - Z: 0.03919 m
- Tilt shaft is aligned with the `CS_Tilt` X-axis
- Perpendicular COG radius: 0.03919 m
- Moment of inertia about tilt shaft: 9.0293e-5 kg*m^2
- Maximum static gravity torque: approximately 0.01948 N*m

The tilt model does not yet include external fasteners, wiring/ribbon cables,
accessory hardware, or omitted rotating horn hardware. These values are a
modeled lower bound. Cable forces may contribute more position-dependent torque
than their mass suggests.

## Preliminary pan-axis properties

These properties are based on the current modeled upper assembly.

- Moving mass about pan axis: 0.11850 kg
- COG relative to `CS_Pan` in the upright/startup pose:
  - X along pan shaft: 0.03269 m
  - Y: 0.00168 m
  - Z: -0.00581 m
- Radial COG offset from pan shaft: approximately 0.00605 m
- Pan inertia in upright/startup pose: 6.54e-5 kg*m^2
- Estimated worst-case pan inertia over current tilt range: 1.69e-4 kg*m^2
- Conservative pan inertia for controller sizing: 1.7e-4 kg*m^2

Assuming the pan shaft is vertical, gravity does not create commanded torque
about the pan axis. Pan torque is dominated by angular acceleration, drivetrain
friction, cable resistance, and any base inclination.

## Current preliminary controller parameters

- Control update target: 60 Hz
- Dynamixel Profile Velocity starting value: 200
- Dynamixel Profile Acceleration starting value: 30
- Tracking smoothness is prioritized over the fastest possible acquisition,
  while retaining enough response to follow a moving target
- Red-dot detection is the current test source; the detector/controller seam is
  intentionally model-neutral for a later object detector

Desired maximum physical pan and tilt speeds have not yet been selected.

## Remaining measurements and validation

- Run the committed controller on the Jetson and collect
  `laser_follow_debug.log`.
- Record stationary, slow-sweep, fast-sweep, reversal, and irregular-motion
  tests when lighting permits.
- Establish the physical aim point in image coordinates rather than assuming
  the numerical image center.
- Measure camera/aiming-axis offsets from the pan and tilt axes.
- Record the missing Dynamixel gains.
- Select desired maximum pan and tilt speeds.
- Perform controlled 50, 100, 250, and 500 pixel step-response tests.
- Record acceptable overshoot and settling time, plus observed backlash,
  buzzing, oscillation, and axis coupling.
- Measure end-to-end camera-to-motion latency before tuning prediction or
  feedforward lead.
- Recalculate final mass, COG, and inertia after all hardware and cable routing
  are complete.
