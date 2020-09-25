
#import sys

from extractors_features.interface_extractors import ExtractorsMain
from clusters.interface_clusters import ClustersMain
#from sequence_generator.interface_sequence_generator import SequenceGeneratorMain
#from sampling.interface_sampling import SamplingMain
#from learning.interface_learning import LearningMain
#from metrics.outputs import OutputMetric

import numpy as np
#import os
#import ast

class Histogram:
	def __init__(self, k, extractor, type_cluster, contrastThreshold, classes):
		self.k = k
		self.type_cluster = type_cluster
		self.contrastThreshold = contrastThreshold
		self.classes = classes
		self.extractor = extractor
		self.cluster_model = None # modelo kmeans

	def set_k(self, k):
		self.k = k

	def get_k(self):
		return self.k

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
	
	def set_type_cluster(self, type_cluster):
		self.type_cluster = type_cluster

	def get_type_cluster(self):
		return self.type_cluster

	

	def create_model(self, descriptor_vstack):

		print('Fazendo o agrupamento')
		cluster = ClustersMain(self.get_k(), self.get_type_cluster(), descriptor_vstack)
		cluster_model = cluster.run() # faz o agrupamento com todos os descritores de todas as classes
		self.set_cluster_model(cluster_model)		


	def extract_keypoints(self, path):
		'''
		param: path onde estão as pastas das classes
		return: primeiro retorno: um dicionario contendo como key cada classe, e o valor de cada key 
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
		#num_imagens = 0
		for cl in self.get_classes():
			path_class = path + '/' + cl
			#num_imagens = num_imagens + len(os.path.exists(path_class))
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

		#self.set_n_images(n_images)

		return dict_class_kp_ds, des_vstack


	def __create_histogram(self, dict_class_kp_ds):

		x_ = []
		y_ = []
		for cl, dict_image_class in dict_class_kp_ds.items():
			for img, kp_ds in dict_image_class.items():
				histogram_image = np.array(np.zeros(self.get_k(), dtype=int))
				for ds in kp_ds[1]:
					visual_word = self.define_cluster_to_point(self.get_cluster_model(), np.array([ds]))
					histogram_image[visual_word] += 1
				x_.append(histogram_image)
				y_.append(self.get_classes().get(cl))

		return x_, y_

				
		'''

        
        self.mega_histogram = np.array([np.zeros(self.get_k(), dtype=int) for i in range(self.get_n_images())])
        old_count = 0

        for cl, dict_image_class in dict_class_kp_ds.items():
			for img, kp_ds in dict_image_class.items():
		        for i in range(self.get_n_images()):
		            l = len(self.descriptor_list[i])
		            if l > self.get_k(): # if key points of the image > alphabet
		                l = self.get_k()
		            for j in range(l):
		                idx = self.kmeans_ret[old_count+j]                    
		                self.mega_histogram[i][idx] += 1
		            old_count += l
		'''


	def define_cluster_to_point(self, model, new_point):	
		'''
		Define the cluster of the new point
		:param new_point: new point
		:param model: cluster model
		:return: cluster of the point
		'''
		
		return model.predict(new_point)[0]

	'''
	def create_graph(self, kps, ds, model_kmeans, weight_dxy, path_img):
	
		labels = model_kmeans.labels_

		for kp, desc in zip(kps, ds):
			point = KeyPoint()
			point.xy = (kp.pt[0],kp.pt[1])
			point.description = desc
			label = self.define_cluster_to_point(model_kmeans, np.array([desc]))			
			point.label = label		
			point.cluster_center = self.get_cluster_centers(label, model_kmeans)
			self.keyPoints.append(point)
	'''

	def run(self, path):
		#path = './train_test' ou './fold1'


		path_ = path + '/train'
		print('extraindo keypoints do train')
		dict_class_kp_ds, descriptor_vstack = self.extract_keypoints(path_)
		self.create_model(descriptor_vstack)
		print('criando histograma do train')		
		x_train, y_train = self.__create_histogram(dict_class_kp_ds)
		

		path_ = path + '/test'
		print('extraindo keypoints do test')
		dict_class_kp_ds, descriptor_vstack = self.extract_keypoints(path_)
		print('criando histograma do test')		
		x_test, y_test = self.__create_histogram(dict_class_kp_ds)

		return x_train, y_train, x_test, y_test


	'''

	def __load_data_set(self, path, histogram):
		
		# read file. prepare file lists.
		images, self.n_images = self.file_helper.getFiles(path)
		
		
		# extract SIFT Features from each image
		label_count = 0
		self.descriptor_list = []
		self.name_dict = {}
		self.train_labels = np.array([])
		self.kp_list = []
				 
		for word, imlist in images.items():
			#self.name_dict[str(label_count)] = word
			self.name_dict[word] = str(label_count)
			# Computing Features for each word
			for im in imlist:
				self.train_labels = np.append(self.train_labels, label_count)				
				kp, des = self.sift.features(im)
				self.descriptor_list.append(des)
				self.kp_list.append(kp)

			label_count += 1
		 
		self.__cluster()
		self.__develop_vocabulary()
	'''


    