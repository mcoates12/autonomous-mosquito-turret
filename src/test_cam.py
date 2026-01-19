import cv2

for i in [0,1]:
	cap = cv2.VideoCapture(i)
	ok, frame = cap.read()
	print(i, "opened:", cap.isOpened(), "frame:", ok, "shape:", None if frame is None else frame.shape)
	cap.release()
