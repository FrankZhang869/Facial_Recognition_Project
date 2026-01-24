from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloader(data_dir="frames", batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(data_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader, dataset.class_to_idx
loader, class_map = get_dataloader()

print(class_map)
for images, labels in loader:
    print(images.shape, labels.shape)
    break
