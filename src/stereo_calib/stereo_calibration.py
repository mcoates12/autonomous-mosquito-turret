import cv2
import numpy as np
import glob
import re

def natsort_key(s):
    return[int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

image_size = None

# Checkerboard settings
pattern_size = (9, 6)           # inner corners
square_size = 0.040             # meters (40 mm)

# Prepare object points
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []      # 3D points
imgpoints_l = []    # 2D left
imgpoints_r = []    # 2D right

left_images  = sorted(glob.glob("left/*.png"), key=natsort_key)
right_images = sorted(glob.glob("right/*.png"), key=natsort_key)

assert len(left_images) == len(right_images)

print(f"Found {len(left_images)} stereo pairs")

for l_img, r_img in zip(left_images, right_images):
    img_l = cv2.imread(l_img)
    img_r = cv2.imread(r_img)

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray_l.shape[::-1]

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size)

    if ret_l and ret_r:
        corners_l = cv2.cornerSubPix(
            gray_l, corners_l, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        corners_r = cv2.cornerSubPix(
            gray_r, corners_r, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

print(f"Usable pairs: {len(objpoints)}")

# Calibrate left camera
ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_l, image_size, None, None
)

# Calibrate right camera
ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_r, image_size, None, None
)

# Stereo calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

ret_s, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    objpoints,
    imgpoints_l,
    imgpoints_r,
    mtx_l,
    dist_l,
    mtx_r,
    dist_r,
    image_size,
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)

baseline = np.linalg.norm(T)

print("\n=== RESULTS ===")
print(f"Left reprojection error : {ret_l:.4f}")
print(f"Right reprojection error: {ret_r:.4f}")
print(f"Stereo reprojection err : {ret_s:.4f}")
print(f"Baseline (meters)      : {baseline:.4f}")
print(f"Baseline (cm)          : {baseline*100:.2f}")

# --- Stereo rectification ---
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l,
    mtx_r, dist_r,
    image_size,
    R, T,
    alpha=0
)

# Generate rectification maps
map1_l, map2_l = cv2.initUndistortRectifyMap(
    mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1
)
map1_r, map2_r = cv2.initUndistortRectifyMap(
    mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1
)

# Save calibration results
np.savez(
    "stereo_calibration_full.npz",
    mtx_l=mtx_l,
    dist_l=dist_l,
    mtx_r=mtx_r,
    dist_r=dist_r,
    R=R,
    T=T,
    R1=R1, R2=R2,
    P1=P1, P2=P2,
    Q=Q,
    map1_l=map1_l, map2_l=map2_l,
    map1_r=map1_r, map2_r=map2_r,
    image_size=image_size,
    square_size=square_size
)

print("\nCalibration + rectification saved to stereo_calibration_full.npz")
