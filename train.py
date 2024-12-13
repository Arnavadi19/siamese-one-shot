import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Siamese
from dataset import OmniglotTrain
from config import TRAIN_DATASET_PATH, MODEL_SAVE_PATH, DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS

def train():
    # Load dataset
    train_dataset = OmniglotTrain(TRAIN_DATASET_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, optimizer, scheduler
    model = Siamese().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()

