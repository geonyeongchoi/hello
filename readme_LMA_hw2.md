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

`python code/run_mlp.py cnn --feature_dir data/cnn --num_features 512`

After downloading the data, I used mlp.py to train the model. 
I used mean values of the output of frames. 

<img width="418" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/af618a19-75db-437b-af63-c97240c7f100">


# 3D CNN Feature Extraction System
First, I tried to use r3d_18 to train the model.However, after training the model, I concluded that the feature does not have enough information to surpass the baseline stably (acc <=95%). Thus, I used the most recent model that is available in torchvision, which is swin transformer. (with Swin3D_B_Weights.Swin3D_B_Weights, DEFAULT)
<img width="690" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/c93c8172-4e80-498e-aa5b-8ce461b3e315">

`swin_extraction.ipynb`

Since the model require a lot of memory, and high CUDA version of GPU (above 11.6 for torchvision 0.16), I decided to use colab to extract features. 

`swin_to_mlp_data_preprocessing.ipynb`

After extracting the features, I modify the structrue of the data that is suitable for the input for mlp.py. 


## Results

After submitting test cases, I found that high batch size (1024) and low test set size is helpful for train the model. 
<img width="561" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/d3f754ec-b79f-46d9-bdff-cb7944b79baf">
Also, having high dropout rate can also boost the performance of the model. Thus, the final structure of our model is shown as follows. 

<img width="602" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/e0842e25-bfe2-4ceb-9c1d-8114b6d40f57">

`version_all.ipynb`

After using features from swin transformer, I can surpass the baseline easily (acc <97%), and I used ensemble method for stable result. Even though using mode values from 200 predictions does not improve the model performances, I expect that this result is more stable than a single result because of the majority voting, so I can get a stable result with the private result too.

