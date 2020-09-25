
import sys

from extractors_features.interface_extractors import ExtractorsMain
from clusters.interface_clusters import ClustersMain
from sequence_generator.interface_sequence_generator import SequenceGeneratorMain
from sampling.interface_sampling import SamplingMain
from learning.interface_learning import LearningMain
from metrics.outputs import OutputMetric

import numpy as np
import os
import ast

class Sequence:
	def __init__(self, k, extractor, type_cluster, search_type, weight_xy, minimum_size_sequence, 
				 contrastThreshold, classes, manipulate_sequence):
		'''
		param: manipulate_sequence -> 0 -> remove as duplicatas, 1 -> não remove as duplicatas
									  2 -> 
		'''
		self.k = k
		self.type_cluster = type_cluster
		self.search_type = search_type
		self.weight_xy = weight_xy
		self.minimum_size_sequence = minimum_size_sequence
		self.contrastThreshold = contrastThreshold
		self.classes = classes
		self.extractor = extractor
		self.manipulate_sequence = manipulate_sequence
		self.cluster_model = None # modelo kmeans

	def set_k(self, k):
		self.k = k

	def get_k(self):
		return self.k

	def set_manipulate_sequence(self, manipulate_sequence):
		self.manipulate_sequence = manipulate_sequence

	def get_manipulate_sequence(self):
		return self.manipulate_sequence

	def set_cluster_model(self, cluster_model):
		self.cluster_model = cluster_model

	def get_cluster_model(self):
		return self.cluster_model

	def set_extractor(self, extractor):
		self.extractor = extractor

	def get_extractor(self):
		return self.extractor

	def set_classes(self, classes):
		self.classes = classes

	def get_classes(self):
		return self.classes

	def set_contrastThreshold(self, contrastThreshold):
		self.contrastThreshold = contrastThreshold

	def get_contrastThreshold(self):
		return self.contrastThreshold

	def set_minimum_size_sequence(self, minimum_size_sequence):
		self.minimum_size_sequence = minimum_size_sequence

	def get_minimum_size_sequence(self):
		return self.minimum_size_sequence

	def set_type_cluster(self, type_cluster):
		self.type_cluster = type_cluster

	def get_type_cluster(self):
		return self.type_cluster

	def set_search_type(self, search_type):
		self.search_type = search_type

	def get_search_type(self):
		return self.search_type

	def set_weight_xy(self, weight_xy):
		self.weight_xy = weight_xy

	def get_weight_xy(self):
		return self.weight_xy


	def create_model(self, descriptor_vstack):

		print('Fazendo o agrupamento')
		cluster = ClustersMain(self.get_k(), self.get_type_cluster(), descriptor_vstack)
		cluster_model = cluster.run() # faz o agrupamento com todos os descritores de todas as classes
		self.set_cluster_model(cluster_model)		


	def extract_keypoints(self, path):
		'''
		param: path onde estão as pastas das classes
		return: primeiro retorno: um dicionario contendo como key cada classe e o valor de cada key 
					é um dicionario, cujo, a key é a imagem o valor um array com os pontos de interesse 
					na primeira posição e descritores da imagem na segunda
					exemplo:			
					{
						cl1: {img1: [[kp],[ds]], img2: [[kp],[ds]]}
						cl2: {img3: [[kp],[ds]], img4: [[kp],[ds]]}	
					}
				segundo retorno: vstack contendo todos os descritores de todas as imagens do path passado como parâmetro
		'''
		des_vstack = None
		dict_class_kp_ds = {}
		#print('--------',self.get_classes())
		for cl in self.get_classes():
			path_class = path + '/' + cl
			#print(path_class)
			# passa o path da classe e o extrator para extrair os pontos de interesse
			ex = ExtractorsMain(path_class, self.get_extractor(), self.get_contrastThreshold())
			# extrai os descritores e pontos de interesse de uma classe
			# data_images -> um dicionario que representa as imagens da classe. key o número da imagem, 
			# value -> um array, onde a primeira prosiçao é o ponto de interesse da imagem e a segunda os descritores
			descriptor_vstack, data_images = ex.run() 

			# vai empilhando os descritores das imagens de cada classe para fazer o cluster
			if des_vstack is None:
				des_vstack = descriptor_vstack
			else:
				des_vstack = np.vstack((des_vstack,descriptor_vstack))
			
			dict_class_kp_ds[cl] = data_images

			#print(len(des_vstack))
		return dict_class_kp_ds, des_vstack


	def create_sequences(self, path, dict_class_kp_ds):
		'''
		param: path do arquivo train.txt ou test.txt
		param: dict_class_kp_ds 
					um dicionario contendo como key cada classe e o valor de cada key 
					é um dicionario, cujo, a key é a imagem o valor um array com os pontos de interesse 
					na primeira posição e descritores da imagem na segunda
					exemplo:			
					{
						cl1: {img1: [[kp],[ds]], img2: [[kp],[ds]]}
						cl2: {img3: [[kp],[ds]], img4: [[kp],[ds]]}	
					}
		'''
		sequenceMain = SequenceGeneratorMain(self.get_cluster_model(), self.get_search_type(), 
										 self.get_weight_xy(), self.get_minimum_size_sequence())


		#print(path)
		#"[['butterfly'],[['2', '0', '2', '2'],['0', '0', '0']]]"
		# para cada imagem de uma classe extrai as sequencias.
		for cl, dict_image_class in dict_class_kp_ds.items():
			for img, kp_ds in dict_image_class.items():
				tokens = path.split('.')
				path_img = '.' + tokens[1] + '/' + cl + '/' + img				
				sequences = sequenceMain.run(kp_ds[0],kp_ds[1], path_img)
				if len(sequences) > 0: # se for gerada sequencia da imagem de acordo com o tamanho mínimo
					out_file = '[\''+cl+'\'],['
					i = 0
					for seq in sequences:
						if i == 0:
							out_file = out_file + str(seq)
						else:
							out_file = out_file + ',' + str(seq)
						i = i + 1
					out_file = out_file + ']'  + '\n'
					#print 'imagem -> %s - %d sequence', img, len(sequences)
					with open(path, 'a+') as file:
						file.write(out_file)


	def remove_duplicates(self, x_list, y_list):
	
		index_actual = 0
		index_to_remove = []
		for x, y in zip(x_list,y_list):
			i = 0	
			for x_t in x_list:
				if index_actual == i:
					i = i + 1
					continue
				if  ''.join(map(str, x_list[index_actual]))  == ''.join(map(str, x_t)):
					index_to_remove.append(i) # save to index to remove of the list
				i = i + 1

			index_actual = index_actual + 1

		indexes = list(set(index_to_remove))
		
		y_list = y_list.tolist()
		for index in sorted(indexes, reverse=True):
			del x_list[index]
			del y_list[index]

		return x_list, y_list

	
	def read_file_sequences(self, file, larger_sequence):
		'''
		param: file arquivo para leitura das sequências
		param: larger_sequence o tamanho máximo de todas as sequências.
		Lê do arquivo as sequencias e retorna dois arrays: as sequências _x e as classes _y de cada sequência.
		Também retorna o maior tamanho de sequência.
		'''
		classes = self.get_classes()
		#print(len(classes))
		with open(file) as f:
			linesfile = f.readlines()
			
		lines = [x.strip() for x in linesfile]
		x_ = []
		y_ = []	
		for line in lines:
			line_to_array = ast.literal_eval(line)
			cl = classes[line_to_array[0][0]] #pega o id da classe
			for sequence in line_to_array[1]:
				if larger_sequence < len(sequence):
					larger_sequence = len(sequence)

				x_.append(np.asarray([int(i) for i in sequence]))
				c = np.array([str(cl)])
				y_.extend(c)
		y_ = np.asarray([int(i) for i in y_])
		
		return x_, y_, larger_sequence



	def read_sequences(self, path):
		'''
		param: path onde estão as pastas das classes
		'''
		larger_sequence = 0 
		file = path + '/train.txt'
		x_train, y_train, larger_sequence = self.read_file_sequences(file, larger_sequence)
		

		file = path + '/test.txt'
		x_test, y_test, larger_sequence = self.read_file_sequences(file, larger_sequence)


		if self.get_manipulate_sequence() == 0:
			x_train, y_train = self.remove_duplicates(x_train, y_train)
			x_test, y_test = self.remove_duplicates(x_test, y_test)
			larger_sequence = 0
			for seq_train, seq_test in zip(x_train, x_test):
				if larger_sequence < len(seq_train) or larger_sequence < len(seq_test):
					if len(seq_train) > len(seq_test):
						larger_sequence = len(seq_train)
					else:
						larger_sequence = len(seq_test)	

		return x_train, y_train, x_test, y_test, larger_sequence

	def run(self, path):
		#path = './train_test' ou './fold1'

		
		path_ = path + '/train'
		#print(path_)

		print('extraindo keypoints do train')
		
		dict_class_kp_ds, descriptor_vstack = self.extract_keypoints(path_)
		#print(dict_class_kp_ds)
		#exit()
		self.create_model(descriptor_vstack)
		path_ = path + '/train.txt'
		print('criando sequencia do train')
		self.create_sequences(path_, dict_class_kp_ds)

		path_ = path + '/test'
		print('extraindo keypoints do test')
		dict_class_kp_ds, descriptor_vstack = self.extract_keypoints(path_)
		path_ = path + '/test.txt'
		print('criando sequencia do test')
		self.create_sequences(path_, dict_class_kp_ds)
		print('lendo sequencias')

		
		x_train, y_train, x_test, y_test, larger_sequence = self.read_sequences(path)

		# larger_sequence maior sequencia gerada no conjunto de treinamento ou teste.
		return x_train, y_train, x_test, y_test, larger_sequence


