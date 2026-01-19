import cv2
import os

# Change if needed
LEFT_CAM  = 1
RIGHT_CAM = 0

WIDTH  = 1920
HEIGHT = 1200

save_dir_left  = "stereo_calib/left"
save_dir_right = "stereo_calib/right"

os.makedirs(save_dir_left, exist_ok=True)
os.makedirs(save_dir_right, exist_ok=True)

capL = cv2.VideoCapture(LEFT_CAM)
capR = cv2.VideoCapture(RIGHT_CAM)

capL.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

idx = 0

print("SPACE = capture | Q = quit")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Camera read failed")
        break

    combined = cv2.hconcat([frameL, frameR])
    #resize preview for display only
    display_scale = 0.5 #50% size, adjust if needed
    preview = cv2.resize(
	combined,
	(int(combined.shape[1] * display_scale),
	int(combined.shape[0] * display_scale))
    )
    cv2.imshow("Left | Right", preview)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        cv2.imwrite(f"{save_dir_left}/img_{idx:02d}.png", frameL)
        cv2.imwrite(f"{save_dir_right}/img_{idx:02d}.png", frameR)
        print(f"Captured pair {idx}")
        idx += 1

    elif key == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
