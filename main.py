#!/usr/bin/env python
# -*- coding: utf-8 -*-


 
import sys
 
from extractors_features.interface_extractors import ExtractorsMain
from clusters.interface_clusters import ClustersMain
from sequence_generator.interface_sequence_generator import SequenceGeneratorMain
from sampling.interface_sampling import SamplingMain
from learning.interface_learning import LearningMain
from metrics.outputs import OutputMetric
from sequence import Sequence
from histogram import Histogram
 
import numpy as np
import shutil
import os
import ast
from datetime import datetime
from sklearn import metrics
import time



extractor_name = {0:'SIFT', 1:'SURF'}

classifier_name = {0:'LSTM', 1:'BOW_SVM', 2:'KTESTABLE', 3:'ResNet50', 4:'Xception', 7:'VGG16', 5:'VGG19', 
	6:'InceptionV3', 11: 'InceptionResNetV2', 8:'DenseNet201', 9:'NASNetLarge', 10:'NASNetMobile', 11:'GRU',12:'MobileNetV2', 13:'ResNet152'}	 
 

type_cluster_name = {0:'KMEANS', 1: 'HIERARQUICO'}
search_type_name = {0:'Breadth', 1: 'Deep'}
sequence_duplicates_name = {0:'SemDuplicatas', 1: 'ComDuplicatas'}



def metrics(file, classes):

	with open(file) as f:
		linesfile = f.readlines()
		
	lines = [x.strip() for x in linesfile]
	inicio = 7 + len(classes)
	fim = inicio + 1
	line = lines[inicio:fim]
	#line = lines[9:10]
	#line = lines[80:81]
	precision = line[0][18:22]
	recall = line[0][28:32]
	f1 = line[0][38:42]
	return precision, recall, f1
  
def save_results(y_true, y_pred, k, extractor, type_cluster, search_type, weight_xy, minimum_size_sequence, 
				 contrastThreshold, classes, manipulate_sequence, path, classifier, batch_size, fineTuningRate,
				 history_train, history_validation, time):
	'''
	path -> exemplo './database/soybean_diseased_healthy' pah do banco
	'''
	
	#acc, precision, recall, f1 = out_put_metrics.calculate_metrics()	

	weight_xy_text = str(weight_xy).replace('.', '')
	contrastThreshold_text = str(contrastThreshold).replace('.', '')
	batch_size_text = str(batch_size).replace('.', '')
	fineTuningRate_text = str(fineTuningRate).replace('.', '')

	file_name = '/' + classifier_name.get(classifier) + '_k' + str (k) +  '_cluster' + type_cluster_name.get(type_cluster) 
	file_name = file_name + '_xy' + weight_xy_text + '_minseq' + str(minimum_size_sequence)
	file_name = file_name + '_st' + search_type_name.get(search_type) + '_dupli'+sequence_duplicates_name.get(manipulate_sequence) 
	file_name = file_name + '_'+extractor_name.get(extractor)+ '_contrast' + contrastThreshold_text 
	file_name = file_name + '_batch' + batch_size_text + '_fineTuning' + fineTuningRate_text 

	file_summary_txt = path + file_name + '.txt'

	out_put_metrics = OutputMetric()

	
	acc = 0
	precision = 0
	recall = 0 
	f1 = 0 
	if sampling == 0:
		acc, precision, recall, f1 = out_put_metrics.calculate_metrics_cross(y_true, y_pred, classes, time)
		out_put_metrics.set_history_train(history_train)
		out_put_metrics.set_history_validation(history_validation)
		out_put_metrics.save_summary_cross(file_summary_txt, classes)
		print('Passou aqui')
	else:
		out_put_metrics.set_y_true(y_true)
		out_put_metrics.set_y_pred(y_pred)
		out_put_metrics.set_history_train(history_train)
		out_put_metrics.set_history_validation(history_validation)
		acc, precision, recall, f1 = out_put_metrics.calculate_metrics(classes)
		out_put_metrics.save_summary_train_test(file_summary_txt, classes)
		

	


	#precision, recall, f1 = metrics(file_summary_txt, classes)
	#acc = out_put_metrics.accuracy()

	line = classifier_name.get(classifier) + ';' + str (k) + ';' + type_cluster_name.get(type_cluster) 
	line = line + ';' + str(weight_xy) + ';' + str(minimum_size_sequence)
	line = line + ';' + str(batch_size) + ';' + str(fineTuningRate)
	line = line + ';' + search_type_name.get(search_type) + ';' + sequence_duplicates_name.get(manipulate_sequence) 
	line = line + ';' + extractor_name.get(extractor) + ';' + str(contrastThreshold)  
	line = line + ';' + str(acc) + ';'+ str(precision) + ';'+ str(recall) + ';'+ str(f1) + ';'+ file_summary_txt +'\n'


	
	file_csv = path + '/results.csv'
	out_put_metrics.line_result_to_csv(file_csv, line)		


def remove_train_test_files(file):
	if os.path.isfile(file):
		os.remove(file)


def run_classifier(k, classifier,x_train, y_train, x_test, y_test, classes_names, path_out, larger_sequence,
					lstm_memory, nb_epoch):
	learning = LearningMain(classifier, x_train, y_train, x_test, y_test, k, None)
	
	path = path_out + "/checkpoints.h5"
	pred, cl = learning.run(nb_epoch, lstm_memory, larger_sequence, path, 0, None, 0)
	
	remove_train_test_files(path)
	path = path_out + "/train.txt"
	remove_train_test_files(path)
	path = path_out + "/test.txt"
	remove_train_test_files(path)
	
	return pred, cl

def experimenter_ktestable(sm, k, extractor, type_cluster, search_type, weight_xy, minimum_size_sequence, 
				 contrastThreshold, classes, manipulate_sequence, classifier, lstm_memory, nb_epoch):
	
	sequence = Sequence(k, extractor, type_cluster, search_type, weight_xy, minimum_size_sequence, 
				 contrastThreshold, classes, manipulate_sequence)

	y_pred = []
	y_true = []
	t = ''
	if sm.what_sampling() == 'Split':
		# onde o train test foi criado, exemplo: './database/soybean_diseased_healthy/train_test'
		path = sm.get_path_out()
		x_train, y_train, x_test, y_test, larger_sequence = sequence.run(path)

		
		pred, cl = run_classifier(k, classifier, x_train, y_train, x_test, y_test, classes, path, larger_sequence, lstm_memory, nb_epoch)
		y_pred = pred
		y_true = cl
	else:
		for fold in range(1, number_fold + 1):
			# onde o train test foi criado, exemplo: './database/soybean_diseased_healthy/train_test'
			path = sm.get_path_out()
			path = path + '/fold'+str(fold)

			start = time.time()

			x_train, y_train, x_test, y_test, larger_sequence = sequence.run(path)	
			pred, cl = run_classifier(k, classifier, x_train, y_train, x_test, y_test, classes, path, larger_sequence, lstm_memory, nb_epoch)
			
			end = time.time()
			elapsed_time = end - start
			t = print_time('k-testable', elapsed_time)
			
			y_pred.append(pred)
			y_true.append(cl)

	save_results(y_true, y_pred, k, extractor, type_cluster, search_type, weight_xy, minimum_size_sequence, 
				 contrastThreshold, classes, manipulate_sequence, sm.get_path(), classifier,0,0, None, None, t)

	

def experimenter_lstm(sm, k, extractor, type_cluster, search_type, weight_xy, minimum_size_sequence, 
				 contrastThreshold, classes, manipulate_sequence, classifier, lstm_memory, nb_epoch):
	
	sequence = Sequence(k, extractor, type_cluster, search_type, weight_xy, minimum_size_sequence, 
				 contrastThreshold, classes, manipulate_sequence)

	y_pred = []
	y_true = []
	t = ''
	if sm.what_sampling() == 'Split':
		# onde o train test foi criado, exemplo: './database/soybean_diseased_healthy/train_test'
		path = sm.get_path_out()
		x_train, y_train, x_test, y_test, larger_sequence = sequence.run(path)

		
		pred, cl = run_classifier(k, classifier, x_train, y_train, x_test, y_test, classes, path, larger_sequence, lstm_memory, nb_epoch)
		y_pred = pred
		y_true = cl
	else:
		for fold in range(1, number_fold + 1):
			# onde o train test foi criado, exemplo: './database/soybean_diseased_healthy/train_test'
			path = sm.get_path_out()
			path = path + '/fold'+str(fold)


			start = time.time()

			x_train, y_train, x_test, y_test, larger_sequence = sequence.run(path)	
			pred, cl = run_classifier(k, classifier, x_train, y_train, x_test, y_test, classes, path, larger_sequence, lstm_memory, nb_epoch)
			
			end = time.time()
			elapsed_time = end - start
			t = print_time('LSTM', elapsed_time)				

			
			y_pred.append(pred)
			y_true.append(cl)

	save_results(y_true, y_pred, k, extractor, type_cluster, search_type, weight_xy, minimum_size_sequence, 
				 contrastThreshold, classes, manipulate_sequence, sm.get_path(), classifier,0,0, None, None,t)

	
	
def experimenter_bow(sm, k, extractor, contrast, cluster, classes, classifier):

	histogram = Histogram(k, extractor, cluster, contrast, classes)
	y_pred = []
	y_true = []
	t = ''
	if sm.what_sampling() == 'Split':
		# onde o train test foi criado, exemplo: './database/soybean_diseased_healthy/train_test'
		path = sm.get_path_out()
		x_train, y_train, x_test, y_test = histogram.run(path)

		learning = LearningMain(classifier, x_train, y_train, x_test, y_test, k, None)
		y_pred, y_true = learning.run(0, 0, k, path,0, None, 0)
	else:
		for fold in range(1, number_fold + 1):
			# onde o train test foi criado, exemplo: './database/soybean_diseased_healthy/train_test'
			path = sm.get_path_out()
			path = path + '/fold'+str(fold)

			#iniciodt = datetime.now().microsecond
			#print('BOW -- inicio --> ',iniciodt)
			start = time.time()
			
			x_train, y_train, x_test, y_test = histogram.run(path)			
			learning = LearningMain(classifier, x_train, y_train, x_test, y_test, k, None)
			pred, cl = learning.run(0, 0, k, path, 0, None, 0)
			fimdt = datetime.now().microsecond
			
			end = time.time()
			elapsed_time = end - start
			t = print_time('BOW', elapsed_time)				

			
			y_pred.append(pred)
			y_true.append(cl)

	save_results(y_true, y_pred, k, extractor, cluster, 0, 0, 0, contrast, classes, 0, sm.get_path(), classifier,0,0,
		None, None,t)

def print_time(approach, elapsed_time):
	t = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
	return t
	#print(approach + '---> ', t)
	#exit()

def run_experimenter_lstm(sm):
	extractors = [0] # 0 SIFT, 1 Surf
	classifiers = [0] # 0 lstm, 1 SVM, 2 KTestable
	type_cluster = [0] # 0 k-maens , 1 hierarquico 
	#k_size = [40, 48, 56, 64] #44,48,52,64
	k_size = [52, 60]
	

	#k_size = [3] #44,48,52,64
	weight_xys = [0.9] # dado um vetor de pesos cria um grafo de distância espacial. Somente o valor 0 no vetor cria um caminho
	#k_size = [128]
	#weight_xys = [0.3]

	minimum_size_sequences = [-1] # -1 (junta as strings do grafo), caso seja 0, pegara a maior string da imagem
	search_types = [1]  # 0 - Breadth-First Search ou 1 - Deep-First Search.
	lstm_memorys = [100]
	manipulate_sequences = [1] # 0,1 -> 0 remove duplicatas, 1 não. 
	contrastThresholds = [0.04] #0.07, 0.08

	nb_epoch = 100

	classes_global = sm.define_class()
	
	for extractor in extractors:
		for classifier in classifiers:
			for cluster in type_cluster:
				for k in k_size:
					for weight_xy in weight_xys:
						for minimum_size_sequence in minimum_size_sequences:
							for search_type in search_types:
								for lstm_memory in lstm_memorys:
									for manipulate_sequence in manipulate_sequences:
										for contrast in contrastThresholds:

											experimenter_lstm(sm, k, extractor, cluster, search_type, weight_xy, minimum_size_sequence, 
				 										 			contrast, classes_global, manipulate_sequence, classifier, lstm_memory, nb_epoch)
											
def experimenter_gru(sm, k, extractor, type_cluster, search_type, weight_xy, minimum_size_sequence, 
				 contrastThreshold, classes, manipulate_sequence, classifier, lstm_memory, nb_epoch):
	
	sequence = Sequence(k, extractor, type_cluster, search_type, weight_xy, minimum_size_sequence, 
				 contrastThreshold, classes, manipulate_sequence)

	y_pred = []
	y_true = []
	t = ''
	if sm.what_sampling() == 'Split':
		# onde o train test foi criado, exemplo: './database/soybean_diseased_healthy/train_test'
		path = sm.get_path_out()
		x_train, y_train, x_test, y_test, larger_sequence = sequence.run(path)

		
		pred, cl = run_classifier(k, classifier, x_train, y_train, x_test, y_test, classes, path, larger_sequence, lstm_memory, nb_epoch)
		y_pred = pred
		y_true = cl
	else:
		for fold in range(1, number_fold + 1):
			# onde o train test foi criado, exemplo: './database/soybean_diseased_healthy/train_test'
			path = sm.get_path_out()
			path = path + '/fold'+str(fold)


			start = time.time()

			x_train, y_train, x_test, y_test, larger_sequence = sequence.run(path)	
			pred, cl = run_classifier(k, classifier, x_train, y_train, x_test, y_test, classes, path, larger_sequence, lstm_memory, nb_epoch)
			
			end = time.time()
			elapsed_time = end - start
			t = print_time('GRU', elapsed_time)				

			
			y_pred.append(pred)
			y_true.append(cl)

	classifier = 11 # GRU para pegar o nome
	save_results(y_true, y_pred, k, extractor, type_cluster, search_type, weight_xy, minimum_size_sequence, 
				 contrastThreshold, classes, manipulate_sequence, sm.get_path(), classifier,0,0, None, None,t)

def run_experimenter_gru(sm):
	extractors = [0] # 0 SIFT, 1 Surf
	classifiers = [4] # 0 lstm, 1 SVM, 2 KTestable, 4 GRU
	type_cluster = [0] # 0 k-maens , 1 hierarquico 
	#k_size = [44,40,48,52,56,60,64] #44,48,52,64
	k_size = [44]
	

	#k_size = [3] #44,48,52,64
	weight_xys = [0.9] # dado um vetor de pesos cria um grafo de distância espacial. Somente o valor 0 no vetor cria um caminho
	#k_size = [128]
	#weight_xys = [0.3]

	minimum_size_sequences = [-1] # -1 (junta as strings do grafo), caso seja 0, pegara a maior string da imagem
	search_types = [1]  # 0 - Breadth-First Search ou 1 - Deep-First Search.
	lstm_memorys = [100]
	manipulate_sequences = [1] # 0,1 -> 0 remove duplicatas, 1 não. 
	contrastThresholds = [0.04] #0.07, 0.08

	nb_epoch = 100

	classes_global = sm.define_class()
	
	for extractor in extractors:
		for classifier in classifiers:
			for cluster in type_cluster:
				for k in k_size:
					for weight_xy in weight_xys:
						for minimum_size_sequence in minimum_size_sequences:
							for search_type in search_types:
								for lstm_memory in lstm_memorys:
									for manipulate_sequence in manipulate_sequences:
										for contrast in contrastThresholds:

											experimenter_gru(sm, k, extractor, cluster, search_type, weight_xy, minimum_size_sequence, 
				 										 			contrast, classes_global, manipulate_sequence, classifier, lstm_memory, nb_epoch)
			

def run_experimenter_bow(sm):
	extractors = [0] # 0 SIFT, 1 Surf
	type_cluster = [0] # 0 k-maens , 1 hierarquico 
	#k_size = [32,40,44,48,52,56,60] #44,48,52,64
	k_size = [44]
	#k_size = [32, 40, 48, 56, 64]
	contrastThresholds = [0.04] #0.07, 0.08
	classifiers = [1] # 0 lstm, 1 SVM, 2 KTestable
	
	classes_global = sm.define_class()
	
	for extractor in extractors:
		for classifier in classifiers:
			for cluster in type_cluster:
				for k in k_size:
					for contrast in contrastThresholds:
						experimenter_bow(sm, k, extractor, contrast, cluster, classes_global, classifier)


def run_experimenter_cnn(sm):

	'''
	architectures = ["Xception", "VGG16", "VGG19", "ResNet50", "InceptionV3",
		"MobileNet", "InceptionResNetV2", "DenseNet201", "NASNetLarge", "NASNetMobile"]
	'''
	#architectures = ['ResNet50', 'Xception', 'VGG16', 'VGG19', 'InceptionV3', 'InceptionResNetV2', 'DenseNet201','NASNetMobile'] # "Xception"
	#architectures = ['ResNet50', 'Xception'] # "Xception"
	#architectures = ['NASNetLarge'] # "Xception"
	#architectures = ['ResNet50','InceptionV3','MobileNetV2','NASNetMobile']
	architectures = ['ResNet152','InceptionResNetV2', 'DenseNet201']

	

	#batch_sizes = [8, 16, 32]
	batch_sizes = [32]
	#epochs = 50
	fineTuningRates = [2,1,0] # 0: with transfer-learning, 1: with fine-tuning, 2: without transfer learning
	
	classes_global = sm.define_class()
	
	for architecture in architectures:
		print('architecture ---------------------------> ', architecture)
		for batch_size in batch_sizes:
			for fineTuningRate in fineTuningRates:
				experimenter_cnn(sm, architecture, batch_size, fineTuningRate, classes_global)
				#exit()

		
def experimenter_cnn(sm, architecture, batch_size, fineTuningRate, classes):

	

	nb_epoch = 50
	y_pred = []
	y_true = []
	history_train = None
	history_test = None
	t=''

	if sm.what_sampling() == 'Split':
		# onde o train test foi criado, exemplo: './database/soybean_diseased_healthy/train_test'
		path = sm.get_path_out()
		train_dir = path + '/train'
		validation_dir = None # sem validacao

		if vali == True:
			validation_dir = path + '/validation'

		test_dir = path + '/test'
		path_models_checkpoints = path  + "/checkpoints.h5"

		

		learning = LearningMain(3, train_dir, None, test_dir, None, 0, validation_dir)
	
		y_pred, y_true = learning.run(nb_epoch, 0, 0, path_models_checkpoints, batch_size, architecture, fineTuningRate)
		


		#remove_train_test_files(path_models_checkpoints)
		history_train = learning.get_history_train()
		history_validation = learning.get_history_validation()

	else:
		learning = None
		for fold in range(1, number_fold + 1):
			# onde o train test foi criado, exemplo: './database/soybean_diseased_healthy/train_test'
			path = sm.get_path_out()
			path = path + '/fold'+str(fold)

			train_dir = path + '/train'
			validation_dir = path + '/validation'
			test_dir = path + '/test'
			path_models_checkpoints = path  + "/checkpoints.h5"

			if vali == False:
				validation_dir = None # sem validacao


			learning = LearningMain(3, train_dir, None, test_dir, None, 0, validation_dir)

			start = time.time()

			pred, cl = learning.run(nb_epoch, 0, 0, path_models_checkpoints, batch_size, architecture, fineTuningRate)
			
			end = time.time()
			elapsed_time = end - start
			t = print_time('CNN', elapsed_time)


			y_pred.append(pred)
			y_true.append(cl)
			remove_train_test_files(path_models_checkpoints)
		learning.setAcc(0)
		#learning.setValidation(vali)
		history_train = learning.get_history_train()
		history_validation = learning.get_history_validation()

		
	classifier_name = {'Xception':4, 'ResNet50':3, 'VGG16':7, 'VGG19':5, 'InceptionV3':6, 
		'InceptionResNetV2':11, 'DenseNet201':8, 'NASNetLarge':9, 'NASNetMobile':10,'MobileNetV2':12, 'ResNet152':13}

		 		 

	save_results(y_true, y_pred, 0, 0, 0, 0, 0, 0, 0, classes, 0, sm.get_path(), classifier_name.get(architecture), 
		batch_size, fineTuningRate, history_train, history_validation,t)

def run_sampling(sampling, path_in, validation):
	# 0 cross validation, 0% train, 0% validation, 0% test
	# 1 train test validation

	sm = None
	if sampling == 0:
		#                 cros, train, validation, test,
		if validation == True:
			#print('---------------> ', number_fold)
			sm = SamplingMain( 0,     0,      0.2,        0.2, path_in, number_fold,type_cross)
		else:
			#print('aqui')
			sm = SamplingMain( 0,     0,      0,        0.1, path_in, number_fold,type_cross)
		sm.run() # gera novo banco de imagem com as amostras separadas
	elif sampling == 1:
		sm = SamplingMain(1, 70, 0, 30, path_in, 0,0)
		sm.run() # gera novo banco de imagem com as amostras separadas
	#exit()
	return sm


'''
ESTÁ CONFIGURADO PARA CROSS-VALIDATION
'''
sampling = 0 # 1 para train test ou 0 cross-validation
number_fold = 10 # AQUI VC MUDA A QUANTIDADE DE DOBRAS

 
vali = False # quando usar leave-one, esse parametro deve ser false
type_cross = 0 # 0-> k-fold, 1-> leave-one

if __name__ == '__main__':
	
	#paths = ['/database/2000','/database/3000','/database/4000'] # aqui vc coloca o eu banco. Na pasta "Seu_banco devem estar as patas de cada classe"
	paths = ['/database/COVID2C20K'] 
	for path_in in paths:
		sm = run_sampling(sampling, base_colab+path_in, vali)
		#run_experimenter_gru(sm)
		run_experimenter_cnn(sm)
		#run_experimenter_bow(sm)
		#run_experimenter_ktestable(sm)

