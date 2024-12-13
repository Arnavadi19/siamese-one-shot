import torch

# Paths
TRAIN_DATASET_PATH = "./data/omniglot_train"
TEST_DATASET_PATH = "./data/omniglot_test"
MODEL_SAVE_PATH = "./models/siamese.pth"

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

