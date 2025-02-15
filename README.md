# EEGNet
EEGNet: A lightweight convolutional neural network for EEG signal classification using PyTorch. This repository includes the model architecture and a training pipeline for efficient EEG signal processing.

# EEGNet: A Compact CNN for EEG Signal Processing

This repository contains an implementation of EEGNet, a lightweight convolutional neural network designed for EEG (electroencephalography) signal classification. The architecture efficiently extracts temporal and spatial features from EEG signals and is optimized for real-time processing.

## Files
- **eegnet.py**: Defines the EEGNet model architecture using PyTorch.
- **trainEEGNet.py**: Contains the training pipeline for EEGNet, including dataset loading, training, and evaluation.

## Features
- **Compact and Efficient**: Uses depthwise and pointwise convolutions for efficient feature extraction.
- **Flexible Architecture**: Customizable number of filters, channels, and pooling parameters.
- **Classification Ready**: Outputs predictions for EEG signal classification tasks with multiple classes.

## Installation
Ensure you have Python 3.8+ and install the required dependencies:
```sh
pip install torch torchvision numpy
```

## Usage
### Training the Model
Run the training script to train EEGNet on synthetic or real EEG data:
```sh
python trainEEGNet.py
```

### Modifying Parameters
You can modify model parameters like the number of filters, channels, and pooling factors in `eegnet.py` and dataset configurations in `trainEEGNet.py`.

## Dataset
The training script uses a synthetic EEG dataset. Replace the `EEGDataset` class in `trainEEGNet.py` with actual EEG data for real-world applications.

## License
This project is licensed under the MIT License.

## Author
Developed by Abhishek Mishra


