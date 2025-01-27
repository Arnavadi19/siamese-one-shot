# Siamese-One-Shot

This repository contains a re-implementation of the seminal paper ["Siamese Neural Networks for One-shot Image Recognition"](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) by Koch et al. The implementation strives to remain as faithful as possible to the methodology and results presented in the paper.

## Key Features

- Reproduces the architecture and approach detailed in the original paper.
- Achieves **92% accuracy**, consistent with the results reported in the paper.
- Implements modern optimization techniques for improved training.

## Deviations from the Original Paper

While maintaining the core methodology, the following modifications have been made:
- **Learning Rate Adjustment:** Used cosine annealing learning rate scheduling instead of learning rate decay.
- **Momentum:** Momentum has not been implemented in this version.

## File Structure

```
Siamese-One-Shot/
├── config.py          # Configuration settings (paths, hyperparameters, etc.)
├── dataset.py         # Dataset classes for training and testing
├── evaluate.py        # Evaluation logic
├── model.py           # Siamese neural network architecture
├── train.py           # Training loop
├── utils.py           # Helper functions (e.g., data transformations)
├── run.py             # Entry point to train or evaluate the model
├── data/              # Directory to store datasets
├── models/            # Directory to save trained models
└── requirements.txt   # Dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Siamese-one-shot.git
   cd Siamese-one-shot
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Omniglot dataset and place it in the `data/` directory. The dataset should have subdirectories for training and testing.

## Usage

### Train the Model

To train the model, run:
```bash
python run.py --mode train
```

The model will be trained using the configuration specified in `config.py`. Checkpoints will be saved in the `models/` directory.

### Evaluate the Model

To evaluate the model on the test dataset, run:
```bash
python run.py --mode evaluate
```

The evaluation script loads the model from the path specified in `config.py` and computes the accuracy on the test dataset.

## Results

This implementation achieves **92% accuracy** on the benchmark dataset, consistent with the performance reported in the original paper.

## References

This implementation has drawn inspiration and guidance from the following repositories:
- [fangpin/siamese-pytorch](https://github.com/fangpin/siamese-pytorch)
- [bhiziroglu/Siamese-Neural-Networks](https://github.com/bhiziroglu/Siamese-Neural-Networks/tree/master)

## Citation

If you find this repository useful, please consider citing the original paper:
```
@article{koch2015siamese,
  title={Siamese Neural Networks for One-shot Image Recognition},
  author={Koch, Gregory and Zemel, Richard and Salakhutdinov, Ruslan},
  journal={ICML Deep Learning Workshop},
  year={2015}
}
```

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
