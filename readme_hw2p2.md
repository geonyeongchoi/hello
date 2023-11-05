## Model Designing Abstracts
I realized the significance of data transformation in enhancing model generalization based on my observations from the early submission experiments, where train accuracy quickly reached 1, whereas validation accuracy remained below 0.6.

# Data Transformation
For data transformation, I decided to apply complex transformations and train a model for an extended period, and then reload the model and apply simpler transformations. These transformations are labeled as "Transform 1," "Transform 2," etc., in the code. Specifically, I trained the model for 600 epochs using the first set of transformations, followed by 200, 100, and 100 epochs for subsequent sets of transformations.
I combined both train and validation set at the last set of epoch.
  
# Model Architecture
I utilized the ConvNext structure for the model architecture, incorporating both 3x3 and 1x1 convolutions to enhance the model's generalization ability and increase the receptive field. The total number of parameters in the model was adjusted to be around 20.5M.
During the initial 600 epochs, I used dropout within the CNN, but it was removed for subsequent training phases.

# Label Smoothing
To prevent the model from being too overconfident in its predictions, I implemented and applied label smoothing.

# Optimizer
For the optimizer, I experimented with AdamW and Adam with small learning rates, but found that they converged slower and tended to overfit compared to SGD. Therefore, I opted not to use them. For the scheduler, I used CosineAnnealingWarmRestarts with a maximum epoch of 100.

# Training epochs
After some initial trials, I figured out that increasing generalization of the model is very important to incresae the performance of the model. Hence, I trained the model with the following 4 transformations having 600, 200, 100, and 100 epochs. 
<img width="989" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/c66b6683-dac0-4206-af53-792ee27ac4f3">
Also, I found that I do not use validation dataset for early stopping since the model converges very slowly, from the 3rd transformations, I combined both training and the validation dataset to train the model.
<img width="583" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/e5d0c3dc-63a4-4f48-b5e9-07f3b6cf2779">
After Combining both training and validation dataset, I can get better results. 

