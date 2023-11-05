# Extracting Sound features 

To extract the sound features, I used mp3 file that I extracted in the HW1 following the instruction in HW1. 

`sound_extract.ipynb`

I used PaSST model to extract the sound feature.

<img width="478" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/8e3240a3-8fce-4b47-b7e6-7c2bc00a0525">

Since there are no mp3 files for 3 data, so I skipped all the data for training the dataset. 

<img width="685" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/6317be6d-bf2a-4461-80a9-2755ba7ac5dc">

# Extracting Video features

To extract the video features I used one of the best pretrained model available in the pytorch called swin tranformer.

<img width="730" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/3db02241-09f0-41cc-93fb-4006807fbafe">

`swin_extraction.ipynb`

To fuse the result, I used two different swin transformer (swin3d_s (small) and  swin3d_b (base)).

<img width="248" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/07910818-ea01-407b-a6c1-defd1696cc11">

I extracted the video feature in the google colab. 

# Early Fusion 

I concatenate sound, swin_b and swin_s features all together, and make a prediction.
(200, 1024, 768 features for each)

`HW3_early_fusion.ipynb`

# Late Fusion

`HW3_late_fusion.ipynb`

"Late fusion" refers to the method of combining the results of two models to obtain a final output. This approach is used to make better predictions or decisions by integrating the final outputs of two models. It is typically implemented in one of the following ways:

Feature-level fusion: The outputs of the two models are transformed into feature vectors, and these feature vectors are combined to create a new feature vector. This new feature vector is then used to generate the final output. For example, in image classification, the outputs of two models are represented as probability distributions, and these probability distributions are combined to determine the final class.

Decision-level fusion: The final outputs or decisions of the two models are directly combined to make the final decision. This is primarily used in classification problems where both models output class labels or binary decisions. For instance, the class labels predicted by the two models can be combined using majority voting to select the final class.

I used Decision-level fusion and use the average value of final decisions of the model. 

<img width="857" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/4073f985-9b28-4102-a2fb-7d289cb33011">

For the late fusion, I predicted the class with different features separately (Sound, swin_b, swin_s). Then, I used average value as for the final score. 


# Double Fusion

`HW3_double_fusion.ipynb`

I extracted features from PaSST, swin_b, and swin_s. Since there are explicit sound information on the features from swin transformer,
I combined features from the sound and swin transformer. Also, I used features from the swin transformer only.
Hence, I used 5 different combinations of the features (1. PasST + swin_b + swin_s 2. PasST + swin_s 3. PasST + swin_b 4. swin_b 5. swin_s).

For the double fusion, I used average value as for the final score. 

# Questions in the handout

## 1. Describe the fusion schemes you choose to implement and the features to fuse. Does fusion
improve the result?

## Early Fusion

In early fusion, I concatenate the features from sound, swin_b, and swin_s to create a single feature vector that represents all modalities. Each feature type is quite large (with dimensions of 200, 1024, and 768), leading to a very high-dimensional input space for the classifier.

Potential Advantages: This approach can enable the classifier to leverage interactions between features from different modalities right from the start of the model.

Potential Disadvantages: The high dimensionality could lead to overfitting if not handled properly. Also, all features are treated equally without considering their unique characteristics or relevance.

## Late Fusion

Late fusion involves combining the decisions of separate models by averaging their predicted probabilities or scores. You've predicted classes using different features (Sound, swin_b, swin_s) independently, and then averaged those values for the final score.

Potential Advantages: This approach reduces the risk of overfitting as each model is trained independently. It also allows for the individual strengths of each model to be utilized, potentially improving robustness.

Potential Disadvantages: It may not capture inter-modal dependencies as effectively as early fusion since the models are combined only at the decision level.

## Double Fusion

Double fusion is a hybrid approach where I combine features from sound and swin transformers in various configurations, and then use averaging for the final decision, similar to late fusion.

Potential Advantages: This approach allows for both intra-modal feature combination (such as combining different swin features) and inter-modal feature combination (such as combining sound with swin features), potentially capturing a broad spectrum of patterns.

Potential Disadvantages: There could be a risk of feature redundancy, especially when similar features are combined, which could potentially dilute significant patterns or lead to overfitting.

## Does fusion improve the result?

No. Through the experiments, I found that swin_s is the best features for the classification.
(validation accuracy: Sound: 65% , swin_b: 99.1% , swin_s: 99.3%). 
After I submitted the final result, predictions with swin_s features have better performance than using different fusion techniques.
(output3.csv)

<img width="987" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/dc2ed80a-b67c-4da2-a98f-d043bf25236a">


It's important to note that the outcome of fusion methods can greatly depend on how complementary the combined features or models are. Fusion tends to be most effective when the individual components capture different aspects of the data and make different kinds of errors, leading to a more robust combined prediction.

In your case, it seems that the swin_s features were strong enough on their own, and combining them with other features didn't lead to any significant improvement. This suggests that swin_s was able to capture the most relevant information for the classification task at hand. It could also indicate that the other features or models were not providing additional complementary information that was helpful for this specific task.

## 2. Report the confusion matrix for multi-class classification in your validation set.

The confusion matrix for multi-class classification having 99.2% and 98% acc in valid and test dataset.
<img width="975" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/1eae3666-64d1-487e-82e9-64c780c9cc7d">

Confusion matrix

<img width="548" alt="image" src="https://github.com/geonyeongchoi/hello/assets/76516262/469d10df-66bd-43a1-baa9-e5e5fd5f239f">

The classes are fairly balanced in terms of the number of instances, as indicated by the relatively uniform numbers along the diagonal. It is hard to see the misclassifying trends of the model since the accuracy is too high.

## 3. Report the time your MED system takes for feature extraction and classification on the testing
set.

For whole datasets,
to extract the video features with Swin tranformer, it takes 8 hours (each), and  
to extract the sound features with PaSST, it takes 30 minutes.

For feature extraction of test data, it takes 1/10 of total time. 

For classification, I used Decision-level fusion (as noted in the handout "In the testing phase, the prediction scores from different models are
then combined to yield a final score."), so classification time for the test set takes less than a second (having average value for the prediction, and take the argmax value from all classes).
