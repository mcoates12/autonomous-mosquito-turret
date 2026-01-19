#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time


def find_laser_centroid_red(
    frame_bgr,
    v_thresh=160,
    s_thresh=120,
    min_area=2,
    max_area=1500
):
    """
    Robust red-laser detection:
    - HSV threshold for red (wraps around hue 0)
    - Requires reasonably high saturation + brightness to avoid white highlights
    - Cleans mask and finds best contour centroid
    Returns: (cx, cy) or None, plus mask for debugging.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Red wraps around hue=0, so use two ranges.
    lower1 = np.array([0,   s_thresh, v_thresh], dtype=np.uint8)
    upper1 = np.array([10,  255,      255],      dtype=np.uint8)
    lower2 = np.array([170, s_thresh, v_thresh], dtype=np.uint8)
    upper2 = np.array([179, 255,      255],      dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    best = None
    best_score = -1.0

    # pick best by brightness inside contour and compactness
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h2 = cv2.boundingRect(c)
        roi_v = v[y:y+h2, x:x+w]
        mean_v = float(np.mean(roi_v)) if roi_v.size else 0.0
        score = mean_v - 0.03 * area
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        return None, mask

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def main():
    ap = argparse.ArgumentParser(description="Red laser dot tracker (no ML).")
    ap.add_argument("--cam", type=int, default=0, help="Camera index for OpenCV VideoCapture")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--v_thresh", type=int, default=160, help="Brightness threshold (V channel) 0-255")
    ap.add_argument("--s_thresh", type=int, default=120, help="Saturation threshold (S channel) 0-255")
    ap.add_argument("--min_area", type=float, default=2, help="Min contour area to consider")
    ap.add_argument("--max_area", type=float, default=1500, help="Max contour area to consider")

    ap.add_argument("--show_mask", action="store_true", help="Show threshold mask window")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try --cam 0/1 or check /dev/video*")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    last_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2

        centroid, mask = find_laser_centroid_red(
            frame,
            v_thresh=args.v_thresh,
            s_thresh=args.s_thresh,
            min_area=args.min_area,
            max_area=args.max_area,
        )

        # draw center crosshair
        cv2.drawMarker(
            frame, (cx0, cy0), (255, 255, 255),
            markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2
        )

        if centroid is not None:
            cx, cy = centroid
            err_x = cx - cx0
            err_y = cy - cy0

            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), 2)
            cv2.putText(
                frame, f"laser: ({cx},{cy})  err: ({err_x},{err_y})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
        else:
            cv2.putText(
                frame, "laser: NOT FOUND (adjust --v_thresh/--s_thresh or lighting)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        # FPS calc
        t = time.time()
        dt = t - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        last_t = t
        cv2.putText(
            frame, f"fps: {fps:.1f}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow("laser_track", frame)
        if args.show_mask:
            cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

