from scipy.spatial import distance

def euc(a, b):
	return distance.euclidean(a,b)

class ScrappyKNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self, row):
		best_dist = euc(row, self.X_train[0])
		best_index = 0
		for x in range(1, len(self.X_train)):
			dist = euc(row, self.X_train[x])
			if dist<best_dist:
				best_dist = dist
				best_index = x
		return self.y_train[best_index]

import random

class RandomClassifier():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = random.choice(self.y_train)
			predictions.append(label)
		return predictions

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5)

#decision tree
#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()

#K-nearest
#from sklearn import neighbors
#my_classifier = neighbors.KNeighborsClassifier()

#Random classifier
#my_classifier = RandomClassifier()

#Scrappy k-nn classifier
my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)
print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)