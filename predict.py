import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from PIL import Image
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def get_pixel_value(*args):
	"""Get pixel value"""
	list_pixel_value = []
	for x in xrange(args[0]):
		for y in xrange(args[1]):
			[value, alpha] = args[2][x, y]
			#list_pixel_value.append((x, y, value)) 
			list_pixel_value.append(value)

	return list_pixel_value

def get_image_size(image):
	"""Get image size"""
	return image.size

def convert_to_grayscale(image_name):
	"""Convert image to grayscale"""
	return Image.open(image_name).convert('LA')

def count_row(filename):
	file_train = open(filename, 'r')
	row_count = sum(1 for row in file_train)
	return row_count

print (count_row('training_data\\train.csv'))
data = pd.read_csv("training_data\\train.csv").as_matrix()
#clf = DecisionTreeClassifier()
#clf = svm.SVC(C=1.0, tol=1e-10, cache_size=600, kernel='linear', gamma=2.0, class_weight='auto')
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50, 10, 5), random_state=2, activation='relu')
clf = svm.SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0, decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
		max_iter=-1, probability=False, random_state=None, shrinking=True,
  		tol=0.001, verbose=False)

#select from 1 to 21000 rows, from column 2 up to end column
xtrain = data[0:318, 1:]
#select from 1 to 21000 rows, from 1st column
train_label = data[0:318, 0]
#print (train_label[len(train_label)-1])

#testing data
xtest = data[318:319:, 1:]
actual_label = data[318:319:, 0]

#fit the data
clf.fit(xtrain, train_label)

print (actual_label[0])
print ('Result',clf.predict([xtest[0]]))

"""Newly added random split"""
#enc = OneHotEncoder(n_values=len(train_label))
#lb = LabelEncoder()
#enc.fit(xtrain)
#lb.fit(train_label)

"""rs = ShuffleSplit(n_splits=2, test_size=.03, random_state=2)
train_index, test_index = rs.split(xtrain).next()
train_X, train_Y = xtrain[train_index], train_label[train_index]
test_X, test_Y = xtrain[test_index], train_label[test_index]
clf.fit(train_X, train_Y)"""

"""Based on input"""
"""img = convert_to_grayscale('to_predict\\predict.png')
im = img.load()
[width, height] = get_image_size(img)	
list_pixel_value = get_pixel_value(width, height, im)
print ('Result',clf.predict([list_pixel_value]))"""

#d = xtest[0]
#d.shape = (10,10)
#plt.imshow(255-d, cmap='gray')

#print (cross_val_score(clf, xtrain, tran_label, scoring='accuracy'))
