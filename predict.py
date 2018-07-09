import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from PIL import Image
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("training_data\\train.csv").as_matrix()
#clf = DecisionTreeClassifier()
#clf = svm.SVC()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2, 1), random_state=2, activation='relu')

#select from 1 to 21000 rows, from column 2 up to end column
xtrain = data[0:8, 1:]
#select from 1 to 21000 rows, from 1st column
tran_label = data[0:8, 0]

#print (xtrain)

#fit the data
clf.fit(xtrain, tran_label)

#testing data
xtest = data[8:9:, 1:]
actual_label = data[8:9:, 0]
print ('To predict',actual_label)

#d = xtest[0]
#d.shape = (10,10)
#plt.imshow(255-d, cmap='gray')
print ('Result',clf.predict([xtest[0]]))

"""Get the accuracy
p = clf.predict(xtest)
count = 0
for i in range(0, 21000):
	count +=1 if p[i] == actual_label[i] else 0
print ("Accuracy ", (count/21000)*100)"""
