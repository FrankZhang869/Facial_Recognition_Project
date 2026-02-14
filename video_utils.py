import cv2
import os
import uuid
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

mtcnn = MTCNN(
    keep_all=False,
    image_size=160,
    post_process=False
)

def extract_frames(video_path, output_root, every_n_frames=1):
    os.makedirs(output_root, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % every_n_frames == 0:

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            face = mtcnn(pil_img)

            if face is not None:
                face = face.permute(1, 2, 0).cpu().numpy()
                face = face.astype("uint8")

                filename = f"{uuid.uuid4()}.jpg"
                frame_path = os.path.join(output_root, filename)

                Image.fromarray(face).save(frame_path)

        count += 1

    cap.release()
