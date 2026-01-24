import torch
from torch import nn, optim
from dataset import get_dataloader
from model import FaceClassifier

def train_model ():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader, class_map = get_dataloader()
    num_classes = len(class_map)

    model = FaceClassifier(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        total_loss = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print("Accuracy:", correct / total)

    torch.save({
        "model_state": model.state_dict(),
        "class_map": class_map
    }, "model.pth")
