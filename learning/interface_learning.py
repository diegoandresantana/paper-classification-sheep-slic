#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from sklearn import metrics
import datetime

if sys.version_info >= (3, 0):
	
	from .lstm import LSTMSyntactic
	from .ktestable import KTestableSyntactic
	from .svm import SVMSyntactic
	from .gru import GRUSyntactic
	
	from .cnn import CNN
else:
	
	from lstm import LSTMSyntactic
	from ktestable import KTestableSyntactic
	from svm import SVMSyntactic
	
	from cnn import CNN
	from gru import GRUSyntactic



'''
Esta classe é uma interface do módulo de geração de sequências.
Ela é responsável por retornar sequências de strings extraídas de imagens. 
'''

class LearningMain:

	def __init__(self, ml, x_train, y_train, x_test, y_test, k, x_validation):
		'''
		:param ml: 0 para LSTM, 1 para SVM, 2 para ktestable, 3 para cnn
		:param x_train: array com os dados de treinamento. Exemplo: [[1,2,2],[1,2,3]]
		:param y_train: array com as classes de treinamento [0,1]
		:param x_test: array com os dados de teste. Exemplo: [[1,2,2],[1,2,3]]
		:param y_test: array com as classes de teste [0,1]
		'''
		self.k = k
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.x_validation = x_validation
		self.ml = ml
		self.acc = 0
		self.history_validation = None
		self.history_train = None
		#self.validation = False

	def getAcc(self):
		return self.acc
		
	def setAcc(self, acc):
		self.acc = acc

	'''
	def getValidation(self):
		return self.validation
		
	def setValidation(self, validation):
		self.validation = validation
	'''
	def get_history_validation(self):
		return self.history_validation
		
	def set_history_validation(self, history_validation):
		self.history_validation = history_validation

	def get_history_train(self):
		return self.history_train
		
	def set_history_train(self, history_train):
		self.history_train = history_train

	
	def run(self, nb_epoch, lstm_memory, larger_sequence,path_models_checkpoints, 
		batch_size, architecture, fineTuningRate):
		'''
		:param nb_epoch: número de épocas
		:param lstm_memory: tamanho da memória da lstm
		:return: 
		'''
		if self.ml == 0:
			print ('Classifier: LSTM')
			lstm = LSTMSyntactic(self.k, self.x_train, self.y_train, self.x_test, self.y_test, nb_epoch, lstm_memory,path_models_checkpoints)
			lstm.pad_sequence(larger_sequence)
			
			lstm.trainModel()
			pred, cl = lstm.testModel() 
			return pred.ravel(), cl

		if self.ml == 4:
			print ('Classifier: GRU')
			lstm = GRUSyntactic(self.k, self.x_train, self.y_train, self.x_test, self.y_test, nb_epoch, lstm_memory,path_models_checkpoints)
			lstm.pad_sequence(larger_sequence)
			
			lstm.trainModel()
			pred, cl = lstm.testModel() 
			return pred.ravel(), cl

		elif self.ml == 1:
			print ('Classifier: SVM')
			svm = SVMSyntactic(self.x_train, self.y_train, self.x_test, self.y_test)
			svm.pad_sequence(larger_sequence)
			svm.trainModel()
			pred, cl = svm.testModel() 
			return pred.ravel(), cl

		elif self.ml == 2:
			print ('Classifier: KTestable')
			ktestable = KTestableSyntactic(self.x_train, self.y_train, self.x_test, self.y_test, self.k)
			ktestable.trainModel()
			pred, cl = ktestable.testModel()			
			return pred.ravel(), cl
		elif self.ml == 3:
			print ('Classifier: CNN')
			train_dir, validation_dir, test_dir = self.x_train, self.x_validation, self.x_test
			cnn = CNN(train_dir, validation_dir, test_dir, batch_size, architecture, nb_epoch,
				fineTuningRate, path_models_checkpoints)
			cnn.trainModel()
			pred, cl = cnn.testModel()

			acc_aux = round(metrics.accuracy_score(cl, pred),2)
			if self.getAcc() < acc_aux:
				self.setAcc(acc_aux)
				self.set_history_train(cnn.get_history_train())
				self.set_history_validation(cnn.get_history_validation())

			return pred.ravel(), cl

		else:
			return None
