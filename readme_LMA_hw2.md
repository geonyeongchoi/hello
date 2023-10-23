# SIFT Feature Extraction System

## Implementation Details
- The system is implemented using Python, utilizing the `pyturbo` library for efficient parallel processing, and OpenCV for SIFT feature extraction.
- The `SIFTFeature` stage is responsible for extracting SIFT features from video frames, while the `BagOfWords` stage converts these features into a bag-of-words representation.
- The detail for running codes follows the github in the handout.

`python code/run_sift.py data/labels/train_val.csv`

`python code/train_kmeans.py data/labels/train_val.csv data/sift 128 sift_128`

`python code/run_bow.py data/labels/train_val.csv sift_128 data/sift`

## Execution Time
- The system is optimized for parallel execution, able to process video data in approximately 1 hour when utilizing 16 CPU cores.
- In a hypothetical scenario with a single CPU core, the execution time would extend to around 16 hours.

## Results
- As a result of the execution, SIFT features of video frames are extracted and represented in a bag-of-words format, ready for further analysis or machine learning applications.

## Code Execution Flow
- The execution begins with parsing command line arguments, specifying input video paths and output directories.
- The `ExtractSIFTFeature` system initializes and starts processing the videos in parallel, extracting SIFT features and saving them in the specified directory.

## Usage of Pyturbo
- The `pyturbo` library is crucial for creating a parallel processing pipeline, with `System`, `Stage`, `Job`, and `Task` classes orchestrating the workflow.
- This ensures efficient utilization of available CPU cores, significantly reducing the time required for processing large amounts of video data.

## Installation and Usage
- Ensure you have Python installed, along with the necessary libraries: OpenCV, NumPy, Pandas, and PyTurbo.
- Run the script from the command line, providing the necessary arguments for input video paths and output directories.

# Running details (random forest)

## Implementation Details
After experimenting with various models such as XGB and MLP, it was observed that there was no significant difference in model performance. Therefore, training was carried out with Random Forest without splitting the dataset into training and validation sets. Eleven Random Forest models, each with 2000 trees (n_estimator set to 2000), were created, and the mode of the results was used for prediction.

## Execution Time
Each model took approximately 10 minutes to train, resulting in a total training time of around 100 minutes.


## Results

`sift_to_RF.ipynb`
The model achieve a classification accuracy > 32.5% on the Kaggle dataset. It utilizes an ensemble of Random Forest classifiers with different random seeds for each iteration to enhance robustness and performance.

# CNN Feature Extraction System

## Implementation Details
This system is designed to extract features from video frames using a Convolutional Neural Network, specifically ResNet18, implemented in PyTorch. Due to issues with GPU utilization, the system currently operates solely on CPU, which results in an execution time of approximately one hour. 

## Execution Time
Originally intended to run on GPU for faster performance, the system had to be adjusted to run on CPU due to unspecified errors. This adjustment has significantly increased the execution time, taking around one hour to complete the feature extraction process.

## Results

`CNN_to_RF.ipynb`

The random forest model structure used earlier was kept intact. Experimental results showed an accuracy of 80% on the test set. I tried to use ResNet34, but there are errors, so if I use ResNet34, then I may have better results.

# 3D CNN Feature Extraction System
I attempted to extract features using the instructions provided on GitHub. However, due to errors during GPU usage and timeout errors during CPU usage (even after adjusting the --job_timeout option), I was unable to extract features using the GitHub code (although I successfully modified the code to work). Therefore, I implemented a custom feature extractor code to extract features from videos.

`3d_custom_extracting.ipynb`

## Results

`3DCNN_to_RF.ipynb`

The random forest model structure used earlier was kept intact. Experimental results showed an accuracy of 80% on the test set.

When extracting features using this code, I exceeded the available GPU memory (11GB), so I extracted features from only a portion of the frames (frames_tensor[:, :, ::5, ::5, ::5]). During extraction, I sampled every fifth frame as well as every fifth width and height. With this data, the prediction performance exceeded an accuracy of 80%, which is suspected to be due to the loss of a significant amount of data during the process of reducing the video's size.
