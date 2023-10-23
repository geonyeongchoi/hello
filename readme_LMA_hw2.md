# SIFT Feature Extraction System

## Implementation Details
- The system is implemented using Python, utilizing the `pyturbo` library for efficient parallel processing, and OpenCV for SIFT feature extraction.
- The `SIFTFeature` stage is responsible for extracting SIFT features from video frames, while the `BagOfWords` stage converts these features into a bag-of-words representation.
- The detail for running codes follows the github in the handout.

python code/run_sift.py data/labels/train_val.csv
python code/train_kmeans.py data/labels/train_val.csv data/sift 128 sift_128
python code/run_bow.py data/labels/train_val.csv sift_128 data/sift

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

# Running details

## Implementation Details
After experimenting with various models such as XGB and MLP, it was observed that there was no significant difference in model performance. Therefore, training was carried out with Random Forest without splitting the dataset into training and validation sets. Eleven Random Forest models, each with 2000 trees (n_estimator set to 2000), were created, and the mode of the results was used for prediction.

## Execution Time
Each model took approximately 10 minutes to train, resulting in a total training time of around 100 minutes.


## Results
The code's objective is to achieve a classification accuracy of at least 32.5% on the Kaggle dataset. It utilizes an ensemble of Random Forest classifiers with different random seeds for each iteration to enhance robustness and performance.

# CNN Feature Extraction System


## Execution Time

## Results

## Code Execution Flow
