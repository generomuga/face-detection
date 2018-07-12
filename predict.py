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

data = pd.read_csv("training_data\\train.csv").as_matrix()
#clf = DecisionTreeClassifier()
clf = svm.SVC()
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 4, 3, 2, 1), random_state=2, activation='relu')

#select from 1 to 21000 rows, from column 2 up to end column
xtrain = data[0:8, 1:]
#select from 1 to 21000 rows, from 1st column
tran_label = data[0:8, 0]

#print (xtrain)

#fit the data
clf.fit(xtrain, tran_label)

#testing data
"""xtest = data[8:9:, 1:]
actual_label = data[8:9:, 0]
print ('To predict',actual_label)"""

img = convert_to_grayscale('to_predict\\input'+str(1)+'.jpg')
im = img.load()
[width, height] = get_image_size(img)
		
list_pixel_value = get_pixel_value(width, height, im)


#d = xtest[0]
#d.shape = (10,10)
#plt.imshow(255-d, cmap='gray')

#print ('Result',clf.predict([xtest[0]]))
print ('Result',clf.predict([list_pixel_value]))
print (cross_val_score(clf, xtrain, tran_label, scoring='accuracy'))