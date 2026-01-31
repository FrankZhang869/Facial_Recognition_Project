from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloader(data_dir="frames", batch_size=32, augment=True):
    if augment:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.ToTensor(),
        ])
    else:
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
