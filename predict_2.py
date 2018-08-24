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
from sklearn.decomposition import PCA
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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



#print (count_row('training_data\\train.csv'))
data = pd.read_csv("training_data\\train.csv").as_matrix()
clf = svm.SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0, decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
		max_iter=-1, probability=False, random_state=None, shrinking=True,
  		tol=0.001, verbose=False)


n_samples, h, w = 319, 100, 100
#select from 1 to 21000 rows, from column 2 up to end column
X = np.array(data[0:, 1:])
n_features = X.shape[1]
#select from 1 to 21000 rows, from 1st column
y = np.array(data[0:, 0])
target_names = np.array(['abi', 'agbada', 'alpante', 'ana', 'atanacio', 'bacho', 'barredo',
			'batoon', 'bismonte', 'brux', 'bryan', 'cabal', 'c1ngan', 'canicosa', 'capuno', 'cariaso', 'carl', 'carlo', 'casipit', 'castro', 'ces',
			'coral', 'cruz', 'cs1', 'cs2', 'cuaresma', 'delapena', 'deramos', 'diaz', 'dina', 'edwards', 'efraim', 'elaine', 'englis', 'erick', 
			'fajardo', 'fule', 'gabby', 'gallares', 'gamilla', 'garcia', 'gem', 'genota', 'glodelyn', 'glorioso', 'golpe', 'gorge', 'greg',
			'hermosa', 'hilbero', 'ibalio', 'IT1', 'IT2', 'iya', 'james', 'jason', 'jc', 'jeux', 'john', 'joseph', 'jp', 'jude', 'kalaw', 'kert', 
			'lani', 'lawas', 'Lizelle', 'lorenzo', 'lou', 'loyola', 'luvim', 'mabalot', 'mane', 'maningo', 'marasigan', 'marife', 'matillano', 
			'matt', 'michaela', 'montes', 'mrbengco', 'palada', 'paolo', 'pineda', 'pulmano', 'quinee', 'reyes', 'rodriguez', 'roselle', 
			'sa', 'sanchez', 'sequitin', 'suarez', 'talatala', 'tan', 'tan2', 'tandayu', 'uy', 'valdez', 'valguna', 'zpredict'])
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

"""Test split"""
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=42)
rs = ShuffleSplit(n_splits=2, test_size=.03, random_state=3)
train_index, test_index = rs.split(X).next()
X_train, y_train = X[train_index], y[train_index]
X_test, y_test = X[test_index], y[test_index]
print (len(X_train), len(X_test), y_test)

n_components = 150
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0), y_pred)

print(classification_report(y_test, y_pred, target_names=target_names))	
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=3):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]-1].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]-1].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

print (y_pred.shape[0])
prediction_titles = [title(y_pred, y_test, target_names, i)
                 for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces
#eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
#plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()