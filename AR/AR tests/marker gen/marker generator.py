import cv2
from cv2 import aruco
import numpy as np

# Define the dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Generate the marker
marker_id = 42  # You can change this ID to any valid marker ID within the dictionary
marker_size = 200  # Size of the marker in pixels

# Draw the marker using the dictionary
marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
marker_image = aruco.drawMarker(aruco_dict, marker_id, marker_size)

# Save the marker image
cv2.imwrite('marker_42.png', marker_image)

# Display the marker image
cv2.imshow('Marker', marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
