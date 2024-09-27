import cv2
import numpy as np
from flask import Flask, render_template, Response, request
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)

def generate_frames(marker_image_stream, overlay_media_stream, media_type):
    # Convert the in-memory files to NumPy arrays
    marker_image_array = np.frombuffer(marker_image_stream.read(), np.uint8)
    overlay_media_array = np.frombuffer(overlay_media_stream.read(), np.uint8)
    
    # Decode the images
    marker_image = cv2.imdecode(marker_image_array, cv2.IMREAD_GRAYSCALE)
    if media_type == "image":
        overlay_media = cv2.imdecode(overlay_media_array, cv2.IMREAD_COLOR)
    elif media_type == "video":
        overlay_media = cv2.VideoCapture(io.BytesIO(overlay_media_array))
    
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)  # Increase features and tune parameters
    kp_marker, des_marker = orb.detectAndCompute(marker_image, None)

    # Use FLANN-based matcher
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = orb.detectAndCompute(gray, None)

        if des_frame is not None:
            matches = flann.knnMatch(des_marker, des_frame, k=2)

            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 10:  # Minimum number of matches to consider a successful detection
                src_pts = np.float32([kp_marker[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                h, w = marker_image.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)

                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

                if matrix is not None:
                    if media_type == "image":
                        warped_media = cv2.warpPerspective(overlay_media, matrix, (frame.shape[1], frame.shape[0]))
                    elif media_type == "video":
                        ret, overlay_frame = overlay_media.read()
                        if not ret:
                            overlay_media.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, overlay_frame = overlay_media.read()
                        warped_media = cv2.warpPerspective(overlay_frame, matrix, (frame.shape[1], frame.shape[0]))

                    mask = np.zeros_like(frame, dtype=np.uint8)
                    cv2.fillConvexPoly(mask, np.int32(dst), (255, 255, 255))
                    masked_frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
                    frame = cv2.add(masked_frame, warped_media)

                    # Add debug text
                    cv2.putText(frame, "Marker Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Add debug text for no marker detection
                cv2.putText(frame, "Marker Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    if media_type == "video":
        overlay_media.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    marker_image = request.files['marker_image']
    overlay_media = request.files['overlay_media']
    media_type = request.form['media_type']

    marker_image_stream = marker_image.stream
    overlay_media_stream = overlay_media.stream

    return render_template('stream.html', marker_image_path=marker_image_stream, overlay_media_path=overlay_media_stream, media_type=media_type)

@app.route('/video_feed')
def video_feed():
    marker_image_stream = request.args.get('marker_image_stream')
    overlay_media_stream = request.args.get('overlay_media_stream')
    media_type = request.args.get('media_type')
    return Response(generate_frames(marker_image_stream, overlay_media_stream, media_type), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
