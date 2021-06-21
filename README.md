# Breast_Cancer_prediction (school_project)
* Built 2 classification models from scratch. the models can help predicting if a patient has cancer or not.
* The data used in this project is a built in dataset in sklearn library.
* Applied 2 models (Logisitc Regression model and K-Nearest Neighbours) and compared their accuracy to find the best model.

## Programming Language & Packages
* Programming Language: Python
* Packages: numpy, pandas, math, sklearn, collections

## the dataset
* Samples total: 569
* features: 30
* classes : 2
  * 0 (bengin)
  * 1 (malignant)

## Models Building
I divding the dataset into train and test sets with train size of 80% and test size of 20%. I build 2 classification models from scratch. some features had values that exceed 1000 and other features have values that is as low as 0.001. Thats mean the range of values are very distinct in each column. The impact of features with higher value will be much higher as compared to the impact of other low valued features. So, scaled all features to common level for a better and more correct prediction. I chose standard scaling, which will make the data in each column between -3 and 3.
* For KNN model: The distances from a sample to all other samples in X_train will be calculated. Then will find the indices of the nearest points/samples (samples with less distance to our sample) and will get the k of them (k is the number of neighbours that we specify). Then will find the class of each sample, and at the end we will get back the most common class, which will be the prediction class for this sample. I calculated the distance between 2 samples using the Euclidean formula.
* For Logisitc Regression model: Gradient descent is used to find the best weights and bias for the model. The model will be trained with the probability of each sample which will be between 0 and 1 (0 is 0% and 1 is 100%), rather than continues values. the probability of each sample is calculated using the sigmoid function.

## Models Performance
The K-Nearest Neighbours out performed the Logisitc Regression:
* Logisitc Regression accuracy in % : 89.473%
* K-Nearest Neighbours accuracy in % : 91.228 %





