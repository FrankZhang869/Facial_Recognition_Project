import cv2
import os
import uuid

def extract_frames(video_path, output_root, every_n_frames=1):
    os.makedirs(output_root, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read() # returns boolean that says if frame has been read, and the frame
        if not ret:
            break

        if count % every_n_frames == 0:
            filename = f"{uuid.uuid4()}.jpg"
            frame_path = os.path.join(output_root, filename)
            cv2.imwrite(frame_path, frame)

        count += 1

    cap.release()
