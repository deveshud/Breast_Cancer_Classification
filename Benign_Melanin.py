
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset

dataset2 = pd.read_csv(r'C:\Users\Devesh Bhardwaj\Desktop\BREAST CANCER PROJECT\data.csv' , header = None)
X2 = dataset2.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]].values
y2 = dataset2.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30 , criterion = 'entropy', random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



print('acc for training data: {:.3f}'.format(classifier.score(X_train,y_train)*100))
print('acc for test data: {:.3f}'.format(classifier.score(X_test,y_test)*100))
