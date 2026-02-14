from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
from video_utils import extract_frames
from cleanup_utils import delete_video_and_frames
from fastapi import Form
from train import train_model
from model import FaceClassifier
import torch
from contextlib import asynccontextmanager
from torchvision import transforms
from PIL import Image
import io
from facenet_pytorch import MTCNN



MODEL_PATH = "model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
class_map = None
idx_to_class = None

mtcnn = MTCNN(
    keep_all=False,
    image_size=160,
    post_process=False,
    device=device
)

def load_model():
    global model, class_map, idx_to_class

    if not os.path.exists(MODEL_PATH):
        print("No trained model found")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_map = checkpoint["class_map"]

    idx_to_class = {v: k for k, v in class_map.items()}

    model = FaceClassifier(len(class_map)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Model loaded for prediction")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()   
    yield

app = FastAPI(lifespan=lifespan)

VIDEO_DIR = "videos"
FRAME_DIR = "frames"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r") as f:
        return f.read()

@app.post("/upload")
async def upload_video(video: UploadFile = File(...), label: str = Form(...)):
    video_id = str(uuid.uuid4())
    filename = f"{video_id}.webm"
    file_path = os.path.join(VIDEO_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await video.read())

    label_dir = os.path.join(FRAME_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    extract_frames(
        video_path=file_path,
        output_root= label_dir,
        every_n_frames=5   # adjust for dataset size
    )

    return {
        "status": "success",
        "video_id" : video_id,
        "label": label
    }
@app.delete("/delete/{video_id}")
async def delete_video(video_id: str):
    delete_video_and_frames(video_id)
    return {"status": "deleted", "video_id": video_id}

@app.post("/train")
async def train():
    train_model()
    load_model()
    return {"status": "training complete"}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not trained yet"}

    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    face = mtcnn(img)

    if face is None:
        return {"error": "No face detected"}

    face = face.permute(1, 2, 0).cpu().numpy().astype("uint8")
    face_pil = Image.fromarray(face)

    x = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        idx_to_class = {v: k for k, v in class_map.items()}
        probs = torch.softmax(outputs, dim=1)[0]

        for idx, prob in enumerate(probs):
            print(f"{idx_to_class[idx]}: {prob.item():.4f}")

        pred_idx = outputs.argmax(dim=1).item()

    label = idx_to_class[pred_idx]
    confidence = torch.softmax(outputs, dim=1)[0][pred_idx].item()

    if confidence < 0.8:
        return {"label": "undefined"}

    return {"label": label}
