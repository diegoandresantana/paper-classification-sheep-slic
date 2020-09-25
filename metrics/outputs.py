#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from sklearn import metrics
from statistics import mean
import sys
import numpy as np
import errno
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

class OutputMetric:
   
	def __init__(self):
		
		self.y_true = None 
		self.y_pred = None

		self.y_true_folds = None 
		self.y_pred_folds = None
		self.history_train = None
		self.history_validation = None

	def set_history_train(self, history_train):
		self.history_train = history_train

	def get_history_train(self):
		return self.history_train

	def set_history_validation(self, history_validation):
		self.history_validation = history_validation

	def get_history_validation(self):
		return self.history_validation

	def set_y_true(self, y_true):
		self.y_true = y_true

	def get_y_true(self):
		return self.y_true

	def set_y_pred(self, y_pred):
		self.y_pred = y_pred

	def get_y_pred(self):
		return self.y_pred

	def save_summary_train_test(self, file_txt, classes):

		acc = round(metrics.accuracy_score(self.y_true, self.y_pred),3)

		out = '****************** Best result ****************\n'	   
		out += '\naccuracy -> ' + str(acc)
		out += '\n\n' + str(metrics.classification_report(self.y_true, self.y_pred, digits=3))	
		
		out += '\n\nclasses_true \n'
		out += '['
		for ch in self.y_true:			
			out += str(ch) + ','
		out += ']'

		out += '\n\npredictions \n'

		out += '['
		for ch in self.y_pred:			
			out += str(ch) + ','
		out += ']'

		out += '\n\nclasses -> ' + str(classes) + '\n\n\n\n\n\n\n'
		fo = open(file_txt, "w")
		fo.write(out)
		fo.close()
		#self.plot_confusion_matrix(classes, file_txt)
		self.plot_batch_size(file_txt)


	def save_summary_cross(self, file_txt, classes):

		acc = round(metrics.accuracy_score(self.y_true, self.y_pred),3)

		out = '****************** Best result ****************\n'	   
		out += '\naccuracy -> ' + str(acc)
		out += '\n\n' + str(metrics.classification_report(self.y_true, self.y_pred, digits=3))	
		
		out += '\n\nclasses_true \n'
		out += '['
		for ch in self.y_true:			
			out += str(ch) + ','
		out += ']'

		out += '\n\npredictions \n'

		out += '['
		for ch in self.y_pred:			
			out += str(ch) + ','
		out += ']'


		accs = []
		precisions = []
		recalls = []
		f1s = []
		
		fold = 1
		str_fold = ''
		for true, pred in zip(self.y_true_folds, self.y_pred_folds):
			result = metrics.classification_report(true, pred, digits=3).splitlines()[len(classes)+4]
			result = result.split(' ')

			str_fold += '\n\n----------------------------------------------------------------------------\nFold -> ' + str(fold)
			str_fold += '\naccuracy -> ' + str(round(metrics.accuracy_score(true, pred),3))
			str_fold += '	  precision -> ' + str(round( metrics.precision_score(true, pred, average='macro') ,3))
			str_fold += '	  recall -> ' + str(round( metrics.recall_score(true, pred, average='macro') ,3))
			str_fold += '	  f1 -> ' + str(round( metrics.f1_score(true, pred, average='macro') ,3))



			str_fold += '\n\nclasses true'
			str_fold += '\n['
			for ch in true:			
				str_fold += str(ch) + ','
			str_fold += ']'

			str_fold += '\n\nclasses predictions'
			str_fold += '\n['
			for ch in pred:			
				str_fold += str(ch) + ','
			str_fold += ']'

			fold = fold + 1

			accs.append(round(metrics.accuracy_score(true, pred),3))
			precisions.append(metrics.precision_score(true, pred, average='macro'))
			recalls.append(metrics.recall_score(true, pred, average='macro'))
			f1s.append(metrics.f1_score(true, pred, average='macro'))
			


		acc = round(mean(accs),3)
		precision = round(mean(precisions),3)
		recall = round(mean(recalls),3)
		f1 = round(mean(f1s),3)
		out += '\n\n----------------------------------------------------------------------------\n'
		out += 'Media geral - ' + str(fold - 1) + ' folds\n'
		out += '\naccuracy -> ' + str(acc)
		out += '	  precision -> ' + str(precision)
		out += '	  recall -> ' + str(recall)
		out += '	  f1 -> ' + str(f1)
		out += '	  standard deviation precision -> ' + format(np.std(precisions), '.2f')
		out += '	  time train one fold -> ' + self.time

		

		

		out += '\n\n----------------------------------------------------------------------------\nFolds...' 

		out += str_fold 

		out += '\n\n----------------------------------------------------------------------------'
		out += '\n\nclasses -> ' + str(classes) + '\n\n\n\n\n\n\n'
		fo = open(file_txt, "w")
		fo.write(out)
		fo.close()

		self.plot_confusion_matrix(classes, file_txt)
		#self.plot_batch_size(file_txt)

	def accuracy(self):
		return round(metrics.accuracy_score(self.y_true, self.y_pred),3)


	def calculate_metrics_cross(self, y_true, y_pred, classes, time):

		self.y_true_folds = y_true 
		self.y_pred_folds = y_pred
		self.time = time

		accs = []
		precisions = []
		recalls = []
		f1s = []
		for true, pred in zip(y_true, y_pred):
			result = metrics.classification_report(true, pred, digits=3)
			result = result.splitlines()[len(classes)+4]
			result = result.split(' ')
			accs.append(metrics.accuracy_score(true, pred))
			#precisions.append(float(result[10])) # versao 9 11
			precisions.append(metrics.precision_score(true, pred, average='macro')) # versao 9 11
			#recalls.append(float(result[15])) # 15 17
			recalls.append(metrics.recall_score(true, pred, average='macro')) # 15 17
			#f1s.append(float(result[20])) # 21 23
			f1s.append(metrics.f1_score(true, pred, average='macro')) # 21 23


		acc = round(mean(accs),3)
		precision = round(mean(precisions),3)
		recall = round(mean(recalls),3)
		f1 = round(mean(f1s),3)

	   

		max_index_precisions = precisions.index(max(precisions))
		self.set_y_pred(y_pred[max_index_precisions])
		self.set_y_true(y_true[max_index_precisions])			 

		return acc, precision, recall, f1



	def calculate_metrics(self, classes):
		acc = round(metrics.accuracy_score(self.y_true, self.y_pred),3)
		result = metrics.classification_report(self.y_true, self.y_pred, digits=3).splitlines()[len(classes)+4]
		result = result.split(' ')
		
		precision = metrics.precision_score(self.y_true, self.y_pred, average='macro')
		recall = metrics.recall_score(self.y_true, self.y_pred, average='macro')
		f1 = metrics.f1_score(self.y_true, self.y_pred, average='macro')

		return acc, precision, recall, f1


	def line_result_to_csv(self, filename, line):

		if not os.path.exists(os.path.dirname(filename)):
			try:
				os.makedirs(os.path.dirname(filename))
			except OSError as exc:
				if exc.errno != errno.EEXIST:
					raise
		with open(filename, 'a+') as file:
			file.write(line)

	def plot_confusion_matrix(self, dict_class, filename):
		s = [(k, dict_class[k]) for k in sorted(dict_class, key=dict_class.get, reverse=False)]
		class_names = []
		for k, v in s:
			class_names.append(k)

		classes = class_names
		cm = confusion_matrix(self.get_y_true(), self.get_y_pred())

		cmap=plt.cm.Blues
		title='Confusion matrix'

		plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes,fontsize=12, rotation=90)
		plt.yticks(tick_marks, classes,fontsize=12)
    

		normalize=False
		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True cannopies')
		plt.xlabel('Predicted cannopies')


		'''
		plt.figure(figsize=(32,32))
		cmap=plt.cm.Blues
		title='Confusion matrix'
		plt.imshow(confusion_matrix_to_print, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		thresh = confusion_matrix_to_print.max() / 2.
		for i, j in itertools.product(range(confusion_matrix_to_print.shape[0]), range(confusion_matrix_to_print.shape[1])):
			plt.text(j, i, format(confusion_matrix_to_print[i, j], 'd'), horizontalalignment="center", 
				color="white" if confusion_matrix_to_print[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		'''

		filename = os.path.splitext(filename)[0]
		plt.savefig(filename + '_matrix.png')



	def plot_batch_size(self, filename):
		
		history = None		
		filename = os.path.splitext(filename)[0]
		filename = filename + '_batch.png'

		plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
		if self.get_history_validation() != None:
			history = self.get_history_validation()
			plt.plot(history.history['val_acc'], label='validation')
		else:
			history = self.get_history_train()

		plt.plot(history.history['acc'], label='train')
		plt.legend()		
		plt.savefig(filename)
