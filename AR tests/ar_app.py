pip install opencv-python-headless numpy flask
import cv2
import numpy as np

# Load the AR marker and the image to overlay
marker_image_path = '/Users/temp/Desktop/fg.png'
overlay_image_path = '/Users/temp/Desktop/ghj.png'
marker_image = cv2.imread(marker_image_path)
overlay_image = cv2.imread(overlay_image_path)

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

# ARUco marker detection setup
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for corner in corners:
            # Overlay the image on detected marker
            pts = corner.reshape((4, 2))
            pts = pts.astype(np.int32)
            src_pts = np.array([[0, 0], [overlay_image.shape[1], 0], 
                                [overlay_image.shape[1], overlay_image.shape[0]], 
                                [0, overlay_image.shape[0]]], dtype=np.float32)
            matrix, _ = cv2.findHomography(src_pts, pts)
            warped_image = cv2.warpPerspective(overlay_image, matrix, (frame.shape[1], frame.shape[0]))
            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, (255, 255, 255))
            masked_frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
            frame = cv2.add(masked_frame, warped_image)

    cv2.imshow('AR Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
