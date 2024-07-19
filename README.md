# Machine Learning Final Project by Faith Villarreal and Ricardo Zapata

## Introduction
**Overview of the project**  
	
 This project addresses the dataset from Spotify's Research and Development team. The aim of this project is to answer a simple but elusive question: will a user skip this song? The dataset, named “Spotify Sequential Skip Prediction Challenge”, is set up such that models can be created to predict whether a user will listen to a track if it is played in some given order. More advanced machine learning algorithms could approximate how sequence and music features play into the outcome. For the purposes of this project, we elected to use the mini dataset provided by Spotify which has approximately 170,000 entries of user session data; the entirety of the larger dataset has a size exceeding 300GB.  The data is split into two separate .csv files: a log of the user's listening session and a log of track feature. The two datasets can be joined based on ‘track_id’: a unique identificator value representing some song. We used the pandas library to merge the two .csv files such that each session track would have both the user info and the track features. The data set contains a multitude of features of which we elected to use 46 features.
This a binary classification problem, with the goal of predicting whether a user will skip the next track played. In this project, we experiment with three different classification models: logistic regression, support vector machines (SVM), and neural networks.

---

## Data Collection and Preparation
**Cleaning The Dataset**  
The raw dataset has no missing values, but did have features that contributed the prediction of whether the track was skipped. 
The variables that contributed to the skip prediction where ‘skip_1’,  ‘skip_2’, ‘skip_3’, and ‘skipped’. We chose to use the result of OR-ing features ‘skip_1’,  ‘skip_2’, ‘skip_3’ as our predicted value for whether a song would be skipped or not. There were several features concerning skipping behavior:
‘skip_1’: Indicates if the track was only played very briefly.
‘skip_2’: Indicates if the track was only played briefly.
‘skip_3’: Indicates if most of the track was played.
‘not_skipped’: Indicates that the track was played in its entirety.
We decided to OR the skip feature booleans in order to focus most simply on our mission with this data set: estimating whether a song will be skipped or not. We hope that this simplification of recognizing a skip (i.e. recognizing any skip as a skip, instead of more granularly how far into a song a song was skipped) will lessen the computational burden as well as offer more accurate estimates for unseen data by decreasing bias.

**Categorical Encoding**  
Many features within the dataset were not float values and thus had to be modified in order for the model to run. For features that had binary outcomes, such as whether the user was using shuffle, or if they had a premium subscription, we encoded the data to 0s and 1s. There were other categories that had much more variety in terms of category. We elected to use one-hot encoding for a large part of these features. For example, some categories described user behavior from the start of the song, indicating if they had reached this song by opening the app, or skipping the last song, or had paused before playing. These categories had zero logical order that would justify ordinal encoding so, despite the costly expansion of the feature columns, we used one-hot encoding. 
	Included in the session log was the date on which a person listened to a song. We elected to break down the date into is year, day, month, and day_of_week components, of which we dropped the day feature.

 **Data Splitting**  
	The data was split into a training set, and a testing set. The training set was 80% of the set and the testing set was 20% of the set.

---

## Model Selection
### Logistic Regression
The first model that our team utilized was a logistic regression model for classifying whether a user would skip the song being played or not. We used Scikit-Learn's LogisticRegression from the linear model module as well as other modules from the Scikit-Learn library to work towards optimizing our logistic regression code. The following is how we worked toward acquiring the best hyperparameters.

**L1 and L2 Regularization**

The first step we took towards optimizing our linear regression code was implementing L1 and L2 regularization. For both L1 and L2, by looping through regularization strengths of 0.001, 0.01, 0.1, 1, 10, we created logistic regression models (using LogisticRegression) and tested their mean-squared error (MSE) and accuracy.

Figure 1: L1 Regularization against Mean Squared Error (MSE) and Accuracy

Figure 2: L2 Regularization against Mean Squared Error (MSE) and Accuracy

After experimenting with L1 and L2 regularization strengths to consider what optimized the logistic regression model best, we used techniques such as standardization, Stratified K-Fold cross validation, and grid search to identify the model with the most optimized hyperparameters. We found that the best parameters were using L2 regularization with a strength of 0.1 and using SelectKBest we were able to select the 10 highest features by k-score. We were able to achieve a ROC-AUC score of 0.984762.

Figure 3: Optimized Logistic Regression Classification Report


Figure 4: Logistic Regression Coefficients (Optimized Model)
Polynomial Feature Transformation
The first type of feature transformation that our team implemented was polynomial feature transformation for our logistic regression model. In order to implement polynomial feature transformation, we utilized the Scikit-Learn library’s built-in preprocessing module which includes PolynomialFeatures. When attempting to use PolynomialFeatures with our data, our team had issues due to our computers not being able to handle the computational complexity required. After data reorganization and encoding, our featurespace had 76 features. Even for just degree 2 polynomial transformation, this means transforming the feature space to sustain computations for 5625–or 76 squared–features. Attempting to run this caused our computers to crash which inspired our pivot to a less computationally taxing feature transformation strategy.


Dataset
Polynomial Transformation Degree


1 (no transformation)
2
Train
0.9791592208720514
N/A
Test
0.9791592208720514
N/A

Table 1: Polynomial transformation results
Square Root Feature Transformation
After being limited in our performance for polynomial feature transformation due to computational constraints, our pivot to square root feature transformation began by starting with the best parameters we already calculated for our original optimized logistic regression model. Making sure to maintain that each respective feature’s value was positive while not inhibiting the integrity of our model was of the utmost importance so we found the minimum of our training and testing data, took its absolute value plus 0.00001, and added this to our training and test data before square rooting. Similarly to our original optimized model, we used SelectKBest to decrease our featurespace and calculated our training and validation accuracies.

Figure 5: Logistic Regression Coefficients (Transformed Feature Model)
Dataset
Square Root Transformation Degree


Original
Transformed 
Train
0.9791592208720514
0.9791666666666666
Test
0.9791592208720514
0.9791666698776302

Table 2: Square Root feature transformation results


---

## Model Training
**Training the selected models**  
Information on training processes and tools used.

---

## Model Evaluation
**Evaluating model performance**  
Includes metrics used for evaluation and results.

---

## Conclusion
**Summary of findings**  
Insights and future work.

---







# Machine Learning Final Project
## by Faith Villarreal and Ricky Zapata



Logistic Regression
The first model that our team utilized was a logistic regression model for classifying whether a user would skip the song being played or not. We used Scikit-Learn's LogisticRegression from the linear model module as well as other modules from the Scikit-Learn library to work towards optimizing our logistic regression code. The following is how we worked toward acquiring the best hyperparameters.
L1 and L2 Regularization
The first step we took towards optimizing our linear regression code was implementing L1 and L2 regularization. For both L1 and L2, by looping through regularization strengths of 0.001, 0.01, 0.1, 1, 10, we created logistic regression models (using LogisticRegression) and tested their mean-squared error (MSE) and accuracy.

Figure 1: L1 Regularization against Mean Squared Error (MSE) and Accuracy

Figure 2: L2 Regularization against Mean Squared Error (MSE) and Accuracy

After experimenting with L1 and L2 regularization strengths to consider what optimized the logistic regression model best, we used techniques such as standardization, Stratified K-Fold cross validation, and grid search to identify the model with the most optimized hyperparameters. We found that the best parameters were using L2 regularization with a strength of 0.1 and using SelectKBest we were able to select the 10 highest features by k-score. We were able to achieve a ROC-AUC score of 0.984762.

Figure 3: Optimized Logistic Regression Classification Report


Figure 4: Logistic Regression Coefficients (Optimized Model)
Polynomial Feature Transformation
The first type of feature transformation that our team implemented was polynomial feature transformation for our logistic regression model. In order to implement polynomial feature transformation, we utilized the Scikit-Learn library’s built-in preprocessing module which includes PolynomialFeatures. When attempting to use PolynomialFeatures with our data, our team had issues due to our computers not being able to handle the computational complexity required. After data reorganization and encoding, our featurespace had 76 features. Even for just degree 2 polynomial transformation, this means transforming the feature space to sustain computations for 5625–or 76 squared–features. Attempting to run this caused our computers to crash which inspired our pivot to a less computationally taxing feature transformation strategy.


Dataset
Polynomial Transformation Degree


1 (no transformation)
2
Train
0.9791592208720514
N/A
Test
0.9791592208720514
N/A

Table 1: Polynomial transformation results
Square Root Feature Transformation
After being limited in our performance for polynomial feature transformation due to computational constraints, our pivot to square root feature transformation began by starting with the best parameters we already calculated for our original optimized logistic regression model. Making sure to maintain that each respective feature’s value was positive while not inhibiting the integrity of our model was of the utmost importance so we found the minimum of our training and testing data, took its absolute value plus 0.00001, and added this to our training and test data before square rooting. Similarly to our original optimized model, we used SelectKBest to decrease our featurespace and calculated our training and validation accuracies.

Figure 5: Logistic Regression Coefficients (Transformed Feature Model)
Dataset
Square Root Transformation Degree


Original
Transformed 
Train
0.9791592208720514
0.9791666666666666
Test
0.9791592208720514
0.9791666698776302

Table 2: Square Root feature transformation results


Support Vector Machines
The second model our team utilized was a support vector machine for classifying whether a user would skip the song being played or not. Our team elected to use the Scikit-Learn python library. We used the linear and polynomial kernel with degree 2, with varying L2 regularization.
Let be noted we did attempt to use a radial basis kernel function but the runtime proved to be nearly impossible to factor into our testing.  For this reason, we also only used a polynomial function of degree 2. 
For the linear model, we tested the L2 values 100, 20, 5, 1, 0.01, and 0.001. 
Linear Kernel
Hyperparameter
C = 1
C = 5
C=20
C = 100
C = 0.01
C= 0.001
Test Accuracy
0.9800
0.9800
0.98
0.9800
0.9801
0.9801
Training Accuracy
0.9794
0.9793
0.9793
0.9793
0.9794
0.9793
Precision
0.9786
0.9786
0.9788
0.9786
0.9785
0.9785
Recall
0.9909
0.9909
0.9910
0.9909
0.9910
0.9909

Table 3: Accuracy of Linear SVM


Figure 4: Linear SVM. C =1.           		Figure 5: Linear SVM. C = 5.

Figure 6: Linear SVM.C = 20. 			Figure 7: Linear SVM.C = 100.

Figure 8: Linear SVM. C = 0.01			Figure 9: Linear SVM. C = 0.001.
Polynomial Kernel
For the polynomial model, we tested the L2 values 100, 20, 5, 1, 0.01, and 0.001. A degree of 2 was used for all testing. The polynomial support vector machine performed poorly for all regularization values. No one regularization brough the accuracy anywhere close to the performance level of the other models. In addition, these reports took the most time to compile which limited our ability to more extensively test regularization techniques. This could be combated in the future with more extensive pruning of the feature being considered. 

  
Figure 10: Polynomial Kernel. C =1. 			Figure 11: Polynomial Kernel. C =5. 

Figure 12:  Polynomial Kernel.C = 20. 	       Figure 13: Polynomial Kernel. C = 100.

Figure 14: Polynomial Kernel. C = 0.001.         Figure 15: Polynomial Kernel. C = 0.01.

Hyperparameter
C = 1.
C = 5
C=20
C = 100
C = 0.01
C= 0.001
Test Accuracy
0.6487
0.6487
0.6487
0.6488
0.6466
0.6487
Training Accuracy
0.6484
0.6416
0.6484
0.6428
0.6442
0.6484
Precision
0.6484
0.6473
0.6512
0.6408
0.6491
0.6484
Recall
0.5435
0.5432
0.5421
0.5433
0.5431
0.5430

Table 4: Accuracy of Polynomial SVM
Neural Network
The third model our team utilized was a neural network for classifying whether a user would skip the song being played or not. Our team elected to use the TensorFlow python library. We tested our model with three and four layers. We tested with ReLu and sigmoid activation functions. We attempted to use sigmoids of more than the last layer and saw dramatic decreases in accuracy for the model, so we did not attempt to vary our activation functions much. For all of our regularization testing, we used ReLu activation for all layers, excluding the last for which we used sigmoid. 
We elected to focus the majority of our testing on a variety of L1 and L2 regularization values, as well as changing the batch size. 
Three-Layer Neural Network
First, a three layer neural network was constructed. The first layer was 64 neurons. The second layer was 32 neurons. And the final layer was 1 neuron from which we can get the predicted class. The activation function of the first two layers was a ReLu function, while the last activation function was a sigmoid function such that the result would be classified. The model was trained with a batch size of 32 and 10 epochs. 
The first, non-regularized, neural network produced a validation accuracy of 98.1%. This accuracy was achieved at the final epoch. 
Next, two L1 values were used: 0.01 and 0.001. L1 produced a lesser accuracy value when set to 0.01. With a value of 0.001, the accuracy did match the basic model with an accuracy of 98.1%.
Then, we tried L2 regularization with 0.01 and 0.001. Both values failed to exceed the basic model’s accuracy. 
Finally, we tried changing the batch size. We used the batch size of 32 for our previous tests. For our two new batch sizes, we used 16 and 64. The batch size of 16 matched the accuracy of the base model. The batch size of 64 produced a marginally worse result. 
Hyperparameter Changed
No Regularization
L1: 0.01
L1: 0.001
L2: 0.01
Batch Size: 64
Batch Size: 16
Training Accuracy
0.9795
0.9719
0.9780
0.9446
0.9772
0.9797
Validation Accuracy
0.9806
0.9697
0.9809
0.9715
0.9789
0.9806
Precision
0.9756
0.9635
0.9765
0.9643
0.9733
0.9767
Recall
0.9959
0.9908
0.9945
0.9929
0.9947
0.9936

Table 5: Accuracy of Three-Layer Neural Network


Figure 16: 3L Neural Network. 		Figure 17: 3L Neural Network. L1 = 0.01.

Figure 18: 3L Neural Network. L1=0.001. 	   Figure 19: 3L Neural Network. Batch = 64. 

Figure 20: 3L Neural Network. Batch = 16. 
Four Layer Neural Network
First, a four layer neural network was constructed. The first layer was 64 neurons. The second layer was 32 neurons. The third layer was 16 neurons. And the final layer was 1 neuron from which we can get the predicted class. The activation function of the first three layers was a ReLu function, while the last activation function was a sigmoid function such that the result would be classified. The model was trained with a batch size of 32 and 10 epochs. 0
The first, non-regularized, neural network produced a validation accuracy of 98.1%. This accuracy was achieved at the final epoch. 
Next, two L1 values were used: 0.01 and 0.001. L1 produced a lesser accuracy value when set to either value. The accuracy was around 98%, so the difference was negligible.
Then, we tried L2 regularization with 0.01 and 0.001. The value of 0.001 matched the basic models accuracy, while the value 0.01 just barely equaled the accuracy of the first. 
Finally, we tried changing the batch size. We used the batch size of 32 for our previous tests. For our two new batch sizes, we used 16 and 64. Neither batch size seemed to dramatically improve the performance of the model, both accuracies hovering right under 98%. 

Hyperparameter
No change
L1 = 0.01
L1 = 0.001
L2 = 0.01
L2 = 0.001
Batch: 64
Batch: 16
Training Accuracy
0.9806
0.9740
0.9675
0.9659
0.9726
0.9592
0.9734
Validation Accuracy
0.9814
0.9795
0.9796
0.9794
0.9804
0.9795
0.9799
Precision
0.9758
0.9781
0.9784
0.9782
0.9779
0.9785
0.9784
Recall
0.9960
0.9906
0.9904
0.9903
0.9921
0.9902
0.9908

Table 6: Accuracy of Four-Layer Neural Network


Figure 21: 4L Neural Network.  		Figure 22: 4L Neural Network. L1 = 0.001.

Figure 23: 4L Neural Network. L1 = 0.01.	Figure 24: 4L Neural Network.L2 = 0.01

Figure 25: 4L Neural Network. L2 = 0.001	     Figure 26: 4L Neural Network.Batch Size = 64

Figure 27: 4L Neural Network.Batch Size = 16


Conclusion / Data Analysis 

Logistic 
 SVM 
 Neural Network
0.9792
0.9801
0.9814

Table 7: Best Accuracy of All Models

Above are the best accuracies achieved by each model. The best performance was by the four layer neural network with no regularization. Excluding the polynomial kernel version of SVM and polynomial feature transformation for logistic regression, our models fit to and perform on our dataset much better than originally expected. 
Based on the accuracy of all the models, the data seemed to be easily linearly separable with no inherent need for feature transformation which helped a great deal in optimization. This was also a great indicator that our models were not underfitting the data from our data which would have resulted in lower validation accuracies due to higher variance. Regularization did not seem to drastically affect the performance of our models, however they did assist us in reducing the complexity of our models and decreasing overall variance without harshly affecting our bias; in terms of accuracy, we observed slight changes, but nothing worth particular note. This may be in part due to the sheer size of the dataset as well as the omission of the session and user ID from our analysis. This omission means we are predicting whether anyone, for any session, would skip a song. This is not a perfect solution to the issue of poor shuffling when listening to music as every person has unique taste in music. The result of our experiment could not be generalized outside of the scope of this specific issue. However, this experiment does lay the groundwork for future research into how taking a more generalized look at song recommendation over a hyper-recommended, user-centric model may actually generalize better over a populace. Through our experimentation with the logistic regression model, we found the user behavior, pertaining to how they arrive on the song, whether from opening the app, shuffling or skipping the last song, was a massive indicator of whether a user would skip a song. This suggests that user behavior, especially with how they interact with that app holds much more sway over the result of their listening rather than the music itself. While specific music may matter to each individual person, our generalized approach to the issue has shown the importance of user-app interaction. 


