import numpy as np
from math import sqrt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Logistic_Regression import LogisticReg
from KNN import KNN

#Breast cancer dataset from sklearn
breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

#split data 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scaling data, using Standarization method which will bring values in a range
# between -3 and 3. We do this for both X_test, and X_train
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# calculate model accuracy.For every sample where y_actual equals y_pred
# increase by one and divide the total by the length of y_actual.
def accuracy_score(y_actual, y_pred):
    return (np.sum(y_actual==y_pred) / len(y_actual))*100

l_reg = LogisticReg(lr=0.0001, num_iters=1000) # calling the class and path learning rate and number of iterations
l_reg.fit(X_train, y_train)
predictions = l_reg.predict(X_test)
print('Logistic regression model accuracy(in %): ', accuracy_score(y_test, predictions))

#calculating suitable k value by taking the sqrt of number of samples in our training dataset "m"
#if the sqrt(m) is even number add 1 to make k odd number to avoid confusion between two classes of data
def calculate_K(X):
    k = 0
    m = X.shape[0]
    if int(sqrt(m)) % 2 == 0: k = int(sqrt(m)) + 1
    k = int(sqrt(m))
    return k

knn = KNN(calculate_K(X_train)) # calling the class and path the calculated k
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
print('KNN model accuracy(in %): ', accuracy_score(y_test, pred_knn))





