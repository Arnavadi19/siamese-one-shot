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

## Installation
To use this repository, clone it and install the required dependencies:

```bash
git clone https://github.com/yourusername/Siamese-one-shot.git
cd Siamese-one-shot
pip install -r requirements.txt
```

## Usage
Train the model on your dataset:

```bash
python train.py --dataset <path_to_dataset>
```

Evaluate the model:

```bash
python evaluate.py --model <path_to_model>
```

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

