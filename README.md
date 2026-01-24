# Webcam-Based Face Recognition Web App

This project is a full-stack machine learning web application that allows users to:
- Record labeled webcam videos for training
- Train a face recognition model locally
- Use the webcam to capture an image and predict the user's identity in real time

The application combines a FastAPI backend, a PyTorch deep learning model, and a browser-based frontend using HTML, CSS, and JavaScript.

---

## Features

- Webcam video recording for dataset collection
- Label-based training workflow
- Face classification using a ResNet18-based PyTorch model
- Live webcam image capture for predictions
- Dynamic UI switching between training and prediction modes
- Local model saving and loading

---

## Tech Stack

### Backend
- Python
- FastAPI
- PyTorch
- Torchvision
- OpenCV (for frame extraction)

### Frontend
- HTML
- CSS
- JavaScript
- MediaDevices API (webcam access)

### Machine Learning
- ResNet18 (PyTorch)
- Supervised image classification
- Custom dataset loader
- Local model training and inference

---

## How It Works

1. **Record Training Data**
   - Users record short webcam videos labeled with their name.
   - Frames are extracted and stored locally.

2. **Train the Model**
   - The model is trained on extracted frames.
   - The trained model and class mapping are saved to disk.

3. **Make Predictions**
   - The user switches to prediction mode.
   - A webcam photo is captured and sent to the backend.
   - The model predicts the most likely identity.

---

## AI Usage

AI-assisted tools were used during the development of this project, primarily to support the creation and refinement of the browser-based frontend UI.

Specifically, AI tools assisted with:
- Structuring and styling the HTML/CSS user interface
- Designing UI transitions between training and prediction modes
- Debugging frontend JavaScript behavior
- Improving usability and visual clarity

All backend logic, machine learning model design, training workflow, and system integration were implemented and fully understood by the author. AI tools were used as a development aid, not as a substitute for core implementation or conceptual understanding.



