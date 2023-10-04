## Geonyeong Choi, Andrew ID: geonyeoc
This code is written and ran at google Colab. 
All you need to do is that put the ipynb file in the google drive and ran it with the colab pro accoun (for memory limitation).
Starting for the medium model architectures in the piazza, I tried to find a architecture that gives good performances. Afterward, I tried to run many epoches from that architecture. I tried spending 140$ for google colab and all the resources, but did not find one. I submitted the model structure that gives me the best result, so the model structure is very simple. There are 120 runs in wandb (https://wandb.ai/geonyeong/hw1p2/table?workspace=user-geonyeoc), but not all experiments is recorded since my colleague told me that wandb ocasionally turns off the colab. 

Explantations related with the code is also written in the markdown.


# Dataset
I used different stratagies for the paddings. 

Case 1. Adding zero padding at the beginning and the end of the MFCCs dataset.
□ ■ ■ ■ ...□
Case 2. Adding zero padding at the end of each frame. 
□ ■ □ ■ ... □ ■ □
Where □ is zero padding and ■ is each frame.

In the case of 2, I have to put dummy labels in the transcript to omit the MFCCs data generated from zero paddings afterwards. 

However, there are no big difference in performances for two cases,
and more time is required for case 2, I did not use it for the last submission.

# Parameters Configuration

I used different hyperparameters (batchsize, context size, and etc. But there are no significant difference in the performance.)

# Training models

I tried to train the datasets with different hyperparameters, and record all the results. 
ex)
for bs in batch_size:
  config['batch_size'] = bs

I trained the model with 30 epochs, and did not use scheduler at first 10 epochs, and use CosineLRWarmupStarts for 20 epochs.

sublists = [random.sample(range(train_data.__len__()), train_data.__len__() // 1) for _ in range(1)]


Also, I divided training datatsets, and recorded the results in the folder /results. There are no difference in recorded results, so I used whole datasets in here. 

# Results 

The predicted results is recorded as a csv file in results folder. I submitted best results with highest validation accuracy. 

