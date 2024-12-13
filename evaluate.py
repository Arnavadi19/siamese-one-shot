import torch
from torch.utils.data import DataLoader
from model import Siamese
from dataset import OmniglotTest
from config import TEST_DATASET_PATH, MODEL_SAVE_PATH, DEVICE, BATCH_SIZE

def evaluate():
    # Load test dataset
    test_dataset = OmniglotTest(TEST_DATASET_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = Siamese().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

            output = model(img1, img2)
            predictions = (output > 0.5).float()
            correct += (predictions == label).sum().item()
            total += label.size(0)

    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate()

