# SpeechDataset and Associated Models for Speech Recognition
This README provides detailed information about the SpeechDataset class and associated models for efficient and advanced speech recognition tasks using PyTorch.
You can run this code if the dataset is located in the right directory. The dataloading cell is commented.
This code is not written in the colab env. If you want to run it in the colab env, you have to modify the code. 
This code uses 460 (360 + 100) dataset without using validation dataset as a training dataset. 

## SpeechDataset Class
The SpeechDataset class is a custom dataset handler, optimized for speech recognition tasks with PyTorch. It efficiently manages speech data, including Mel Frequency Cepstral Coefficients (MFCCs) and transcripts, ensuring minimal memory usage.

## How to Use
Initialize the SpeechDataset class with the root directory of your dataset and the desired partition:

train_dataset = SpeechDataset(
    root='data_directory',
    partition='train-clean-100',
    transforms=your_transforms
)

Key Features
Memory Efficiency: Data is loaded in __getitem__, reducing RAM usage.
Flexibility: Supports different data partitions and transforms.
Batch Processing: Compatible with PyTorch's DataLoader.
Associated Models
Several models are provided for processing and understanding speech data:

## PositionalEncoding
Adds positional encodings to input sequences, enhancing the model's understanding of sequence data.

## TransformerEncoder
A Transformer encoder block with multihead attention and feed-forward networks.

## pBLSTM
Pyramidal Bidirectional LSTM for reducing time resolution of input features and capturing bi-directional context.

## TransformerListener
A combination of pBLSTMs and Transformer encoder layers for processing speech features.

## Attention
Custom attention mechanism for aligning decoder output with encoder feature sequence.

## Speller
Decoder using sequence-to-sequence prediction for speech transcription.

## Training and Validation
Functions train and validate are included for model training and performance evaluation on a validation set.

## Configuration and Execution
To utilize the provided code:

## Set your data directory and configurations (DATA_DIR and config).
Initialize SpeechDataset for training, validation, and testing.
Create model instances, define optimizer, loss function, and learning rate scheduler.
Execute the train function in an epoch loop, with optional validation.
Additional Information
Required libraries include torch, numpy, etc.
Hyperparameters in model initializations can be adjusted as needed.
The code is modular for easy experimentation and customization.
This guide enables efficient training and evaluation of speech recognition models using the provided classes and function
