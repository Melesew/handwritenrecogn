#import required liberaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('train.csv').as_matrix()

clf = DecisionTreeClassifier()

#xtraining dataset
xtrain= data[0:21000, 1: ]
train_label = data[0:21000, 0]
clf.fit(xtrain, train_label)

#testing data
xtest = data[21000: , 1: ]
actual_label = data[21000: , 0]

# take random test and check if it is working
data = xtest[8]
data.shape = (28, 28)  # change row vector to 28 x 28 mtrx 
plt.imshow(255-data, cmap='gray')  # make background white, 255-data
print(clf.predict( [xtest[8]] ))
p = clf.predict(xtest)


# calculate accuracy
count = 0
for i in range(0, 21000):
    count += 1 if p[i] == actual_label[i] else 0
print("accuracy : ", (count / 21000) * 100)

plt.show()