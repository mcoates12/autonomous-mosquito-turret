## Project Update

I was so excited with the progress I made that I logged off and decided to watch a movie and relax. It’s honestly kind of amazing seeing how far this turret has come already, and I’m even more excited to keep pushing forward.

---

## Camera Calibration Progress

E-con finally emailed me back, and I wrote a short OpenCV script to take photos for camera calibration. In total, **22 photos** were taken, with **21 usable** according to OpenCV.

Initially, I tried using my cell phone, but it has a privacy screen that blurs the image edges at an angle, which made it unusable. Since I don’t have access to a printer or a Sharpie, I switched to using my tablet instead.

The first attempt with the tablet was a no-go due to camera lighting settings. After adjusting exposure and brightness on both the cameras and the tablet, I finally got calibration results that (from what I’ve read) are considered quite good:

OpenCV camera errors
Left reprojection error: 0.1703
Right reprojection error: 0.1772
Stereo reprojection error: 0.5314
Baseline (meters): 0.0500
Baseline (centimeters): 5.00


Originally, I generated the checkerboard online with square sizes of **40 mm × 40 mm**, which resulted in an estimated baseline of ~11 cm. That seemed wrong given how close the cameras actually are.

After some checking, thinking, and double-checking, I realized the tablet was **shrinking the image** to fit its aspect ratio. I measured the squares manually with a ruler and found they were actually **18 mm × 18 mm**. After updating that value in the calibration script, OpenCV correctly calculated the camera baseline.

I also added:
- Image rectification (to keep objects aligned on the same axis)
- Centroid detection

---

## Servo Movement

Next, I started working on servo movement. I wrote a small script using the **Dynamixel SDK** and got basic control working.

Unfortunately, I don’t yet have a proper baseplate for the turret, and it’s pretty top-heavy. It kept falling over, so I improvised:
- Broke off a plastic clothes hanger
- Secured it together with electrical tape
- Taped one of my older pan mounts onto it so the servo could sit inside

Not pretty — but functional for now.

---

## Object Detection & Laser Tracking

Once the servos were responding correctly, I started working on actual object-tracking logic. I created `laser_track.py` and decided to test it using a laser pointer on the wall.

Since I have a cat (and cats love laser pointers), it felt like a perfect test case.

Right now, the script:
- Detects the **hue** of the laser
- Initially only detected the laser within ~6 inches of the camera
- Improved detection range by tuning `v_thresh` and `s_thresh`

Lighting was still somewhat challenging due to sunlight coming through my window, but it worked well enough. Watching the turret move and track the laser for the first time was *incredibly* exciting.

After recording a few videos and posting on Facebook and Instagram, I noticed the movement felt slow. I started tuning motion parameters and landed on:

deg_per_px: 0.006 (pan & tilt)
velocity profile: 200
steps: 3
deadband_px: 10


The turret is now significantly faster, but it feels **jittery** — like it wants to move faster but something is holding it back. I expect most of this will be resolved once I integrate proper **PID control**.

---

## Next Steps

Planned work moving forward:

1. Implement **PID controls**
2. Add **prediction logic** using an **Unscented Kalman Filter**
3. Move into AI vision:
   - **Option A:** Find an existing mosquito-detection model
   - **Option B:** Train my own model

I don’t expect GPU limitations to be an issue — I’m running an **RTX 5090**, so local training should be fine. The bigger challenges will be:
- Finding or building a dataset
- Labeling bounding boxes
- Training the model properly

I also plan to include images of:
- Humans
- Cats
- Dogs
- Other insects

This will allow the model to differentiate targets and enforce safety logic so the turret **does not engage** near people or pets.

Additional planned features:
- Safety protocols
- Laser integration
- Audio cues (beeper tones) for:
  - Scanning
  - Lock-on
  - Firing

---

## Project Status & Reflections

It’s hard to say exactly how far along I am, but I’d estimate I’m around **halfway**, maybe a bit more.

This milestone has been one of the most exciting things I’ve ever experienced. Watching something go from a random idea to a system that can **see**, **track**, and **move** is easily one of the top five most rewarding things I’ve ever done.

There are still hardware challenges ahead — especially with the **CSI camera cables**, which are far too short to allow full turret motion. I’m trying to avoid buying new cameras if possible. I may experiment with:
- Short CSI cables → CSI-to-Ethernet converters → Ethernet → back to CSI

I’ve read that longer CSI runs can introduce lag or signal issues, so this will require experimentation.

---

## Closing

Thanks to everyone who’s been following along. I genuinely can’t stress enough how exciting this project has been for me.

See you all next week 🚀
