from sklearn.svm import SVC
from keras.preprocessing import sequence
import numpy as np


class SVMSyntactic:
	def __init__(self, x_train, y_train, x_test, y_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.svclassifier = None

	def pad_sequence(self, size):
		self.x_train = sequence.pad_sequences(self.x_train, maxlen=size)
		self.x_test = sequence.pad_sequences(self.x_test, maxlen=size)

	def trainModel(self):
		self.svclassifier = SVC(kernel='linear')  
		self.svclassifier.fit(self.x_train, self.y_train)

	def testModel(self):

		predictions = self.svclassifier.predict(self.x_test)
		classes = self.y_test
		return (predictions, np.asarray([int(i) for i in classes])) 
