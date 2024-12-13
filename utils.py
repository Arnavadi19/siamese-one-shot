from torchvision import transforms

# Data Transformations
def get_transforms():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

