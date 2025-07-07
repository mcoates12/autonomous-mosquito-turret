# Weekly Project Update – Week 1

This week’s update will be shorter than most. I’m taking two summer classes and had two exams this week, so most of my time went into studying, recovering from an illness, and the holiday. Still, I’m happy with the progress I made over the last few days.

I bought a SolidWorks 2025 student license a few months ago, and a buddy of mine let me borrow his Ender-3 Pro printer. I spent most of the week experimenting with printing and learning how the software and hardware work together.

---

## Mount Design and Modifications

I began by searching for different mounts I could 3D print and prototype with to accelerate my turret design process. Eventually, I stumbled upon a side-mount bracket that I’m currently using/modifying for my own specific use case ([see hardware/3d-models/credits.md for source info](./hardware/3d-models/credits.md)).

I printed off the original bracket and it fit perfectly on my XL430-W250-T servo—such a great match.

I started creating the platform that would attach to my pan servo horn so I connect both the pan and tilt servo by those brackets. I created the `PanMount.SLDPRT` file (in `hardware/3d-models`) using SolidWorks, which was a great learning experience. Rather than keeping an unnecessary part,

I thought: *"Why not just modify the original instead?"*

I imported the actual XL430 SolidWorks file so I could get accurate measurements of the screw holes for the horn and distances between each hole. I ended up modifying the bracket five times.

---

## Iterative Prints and Challenges

The first modification is saved as `dynamixel_mounting_plate_v102.stl`. When I printed it, I noticed the screws wouldn’t fit due to the bracket’s thickness (~5mm). I decided to add indents to the bracket so the screw heads could sit flush and the servo would still be able to fit.

This led to `mountfinal.stl`, which has ~0.8mm diameter holes with indents around 1mm deep. Unfortunately, the screw heads didn’t seat properly. I increased the radius in `mountfinalV2`, but the indents weren’t deep enough and the bracket printed messily.

Eventually, I adjusted the depth to ~3.5mm and the radius to ~2mm, resulting in a clean and well-fitting part: `finalmountV3.stl`. But there was another issue. The middle of the servo horn is where the screw is that attaches the horn to the actual servo and it sits at a 2mm height above where the actual screws are located. Which was something I didn't even think about when modifiying the original bracket.

---

## Slicer Settings and Printing Details

I’m currently printing with:
- Wall thickness: 1 mm  
- Wall line count: 3  
- Top/bottom layers: 6  
- Infill: 50%  
- Ironing: Enabled (top only)  
- Ironing flow: 20%  
- Print temp: 210°C (PLA+)  
- Speed: 30 mm/s depending on pass  
- Layer height: 0.2 mm  
- Supports: None  
- Slicer: Ultimaker Cura  

---

## Bracket Update and Next Steps

To avoid modifying the mount further and waiting another 3 hours for it to print, I looked around at the few hardware stores that are open on a Sunday to see if there was any longer m2.5 screws I could buy. Unfortunately, none of the nearby hardware stores that were open carried M2.5 screws, 

so I designed `finalmountV4.stl`, which uses a 2mm indent to better accommodate the raised screw head in the center of the horn.

so I’m waiting another 3 hours for this thing to finish printing.

This version allows the bracket and horn to sit flush while also providing holes for both the bracket and servo screws.

Next, I’m planning to create a **C-style tilt mount** for the second servo, ensuring enough spacing and proper alignment with the pan bracket. This should allow for synchronized movement.

---

## Looking Ahead

Next week, I’ll be:
- Finishing the bracket for the **tilt servo**
- Testing **servo movement**
- Installing cameras
- Starting work on **YOLOv8 model training**
- Learning **OpenCV** and depth estimation with stereo vision
- Beginning to write modular **Python code** for servo/camera integration

This week’s quote, as I begin this journey:

> *"A journey of a thousand miles begins with a single step."* — Lao Tzu

---
