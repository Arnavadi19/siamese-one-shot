import argparse
from train import train
from evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="Run Siamese Neural Network")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], required=True,
                        help="Specify whether to train or evaluate the model")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "evaluate":
        evaluate()
    else:
        print("Invalid mode selected. Choose 'train' or 'evaluate'.")

if __name__ == "__main__":
    main()

