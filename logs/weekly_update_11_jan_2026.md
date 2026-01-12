# Weekly Update – 11 Jan 2026

Progress is going great! I have finally built the tilt mount correctly. Admittedly, it took three additional iterations before I got it right. Along the way, I learned that M2, M2.5, and M3 screws typically use **Japanese Industrial Standard (JIS)** drivers rather than standard Phillips heads, which has significantly improved assembly.

Earlier this week, I spent a day running around trying to find JIS drivers and other necessary tools. I eventually found a large driver set that included JIS bits, picked up wire cutters, and bought additional screws. I then had to make a few trips back to ACE Hardware to find the correct screw length for the cameras. I ultimately settled on **M2×20 screws** and successfully mounted both cameras.

I downloaded **Dynamixel Wizard 2.0** and set up power distribution through the power hub so the servos were properly powered. Using the Wizard, I configured the servos and set the tilt bounds to **62.23° – 191.78°**, ensuring the mount does not collide with the servo or other parts of the system.

Next, I flashed **JetPack 6.2** onto my Jetson Orin Nano and installed a **2 TB M.2 SSD**. After completing the setup, I attempted to install the camera software from the E‑consystems website. Unfortunately, they only provided **L4T version 36.4.0**, while my system is running **36.4.7**. As a result, I’ve been unable to install the camera drivers, QtCAM, or OpenCV. I emailed E‑consystems yesterday requesting the updated drivers and associated files and am currently waiting for a response.

At the moment, progress is somewhat dependent on E‑consystems, so hopefully they note reply soon. In the meantime, I can begin working on basic servo‑movement logic and decide on an object‑detection model. I’m currently leaning toward **YOLOv11**, but I plan to do more research before making a final decision.

Hope everyone has a great week! I’ll be back next week with another progress update. School starts next week as well, but I’ll be doing my best to maintain momentum and keep pushing this project forward.

