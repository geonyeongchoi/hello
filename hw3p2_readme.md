This code is written in jupyter notebook, and changed to run on colab environment. 
You can run this code by simply running in colab.

To load the dataset more the model, we should put zeros to deal with the different length of the sequences by using pad_sequence.
In collate_fn, we use pad sequence ,considering that the data form of batch is same as the returned form from 'getitem'.

<img width="765" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/045508a9-edcf-4efd-8e96-86705dd3237d">

After putting zero in packed sequences, we decrease the number of sequence (1/2).

<img width="435" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/e05f77a3-6b8c-4126-b5d6-83ff2839bbfd">

In the encoder (LSTM), we pack the padded sequence to decrease the number of calculation. 
<img width="602" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/2627adb6-2d7e-4457-80a7-b776130b8fe1">

The shape of the decoder is a diamond.
<img width="314" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/e904853e-3b3d-4f51-9e58-51871222ab90">

After running few steps, I figured out few hyperparameters are very important. 
<img width="1366" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/45707539-5bcd-47e4-8ba1-7a5c2afc5757">

1. Having 4 layers on BiLSTM (num_layers = 4)
2. If I set too many parameters, the training loss does not decrease. However, before reaching the maximum number of parameters, having more parameters can decrease the validation dist.
3. Dropout rate should be 0.3, and batch normalization is required.


I also found that having good hyperparameters, validation distance on the first epoch should be less than 20 as shown below.

<img width="412" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/693ee3ac-2be9-4377-8406-28dd48657145">

After running more than 60 epochs, I found that validation loss is not converged anymore, so I stopped training at 73 epochs, and submitted final result.

<img width="416" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/4c9c7f7b-3b69-49c0-8d5c-82face1a62dc">

Here is the link for wandb.
wandb link:
https://wandb.ai/geonyeong/hw3p2?workspace=user-

