#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PIL import Image
from random import shuffle
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import numpy as np 
import cv2
import shutil

class Sampling(object):
	def __init__(self, path):
		self.path = path
		self.path_out = None
		self.classes = self.define_class()

	def get_classes(self):
		return self.classes
		
	def set_path_out(self, path):
		self.path_out = path

	def get_path_out(self):
		return self.path_out

	def get_path(self):
		return self.path

	def set_path(self, path):
		self.path = path

	def delete_file(self, file):
		if os.path.exists(file):
			os.remove(file)

	def listdir_nohidden(self,path):
		for f in os.listdir(path):
			if not f.startswith('.'):
				yield f


	def __listDir(self, path):
		if os.path.exists(path):
			return self.listdir_nohidden(path)
		else:
			return None

	def removeDirectory(self, directory):
		if os.path.isdir(directory):
			shutil.rmtree(directory)

	def createDirectory(self, directory):
		if not os.path.exists(directory):
			os.makedirs(directory)

	def readImage(self, path):
		'''
		Read image of the path
		:param path: path to image
		:return: image
		'''
		img = cv2.imread(path)
		return img

	def read_dataset(self):
		'''
		Retorna um dictionary onde key é a classe e o value é uma lista com os nomes de aruivos de imagens
		exemplo: {
					'soja_doente': ['img1.jpg','img2.jpg','img3.jpg'],
					'soja_sadia': ['img4.jpg','img5.jpg','img6.jpg']
				 }
		:param path: path to image
		:return: image
		'''		
		images_dict = {}
		list_class = self.__listDir(self.get_path()) # lista todas as pastas, classes

		print(list_class)
		if list_class is not None:
			for cl in list_class: 
				list_images = []				
				images = self.__listDir(self.get_path() + '/' + cl + '/')
				for img in images: # para cada classe, pega todos os nomes de arquivo de imagens
					list_images.append(self.get_path() + '/' + cl + '/' + img) # adiciona cada nome de arquivo de imagem em uma lista
				images_dict[cl] = list_images # key -> o nome da classe, value -> a lista com o nomes dos arquivos da classe
		return images_dict

	def define_class(self):
		classes = {}
		i = 0
		
		for cl in os.listdir(self.get_path()):
			if not os.path.isfile(self.get_path()+'/'+cl):
				if cl != 'folds' and cl != 'train_test':
					classes[cl] = i
					i = i + 1
		return classes

	def list_image_path(self, path):
		'''
		Retorna um dictionary onde key é a classe e o value é uma lista com os nomes de arquivos de imagens
		exemplo: {
					'soja_doente': ['img1.jpg','img2.jpg','img3.jpg'],
					'soja_sadia': ['img4.jpg','img5.jpg','img6.jpg']
				 }
		:param path: path to image
		:return: image
		'''		
		images_dict = {}
		list_class = self.__listDir(path) # lista todas as pastas, classes
		if list_class is not None:
			for cl in list_class: 
				list_images = []				
				images = self.__listDir(path + '/' + cl + '/')
				for img in images: # para cada classe, pega todos os nomes de arquivo de imagens
					list_images.append(path + '/' + cl + '/' + img) # adiciona cada nome de arquivo de imagem em uma lista
				images_dict[cl] = list_images # key -> o nome da classe, value -> a lista com o nomes dos arquivos da classe
		return images_dict

	def images_train_test(self):

		'''
		return: um array com duas posicoes. Na primeira um dicionario com os dados do treinamento.
				Na segunda um dicionario com os dados do teste.
		Exemplo:
		[
			{'diseased_soybean': ['1.jpg', '3.jpg', '11.jpg'], 'healthy_soybean': ['6.jpg', '12.jpg', '14.jpg']}, 
			{'diseased_soybean': ['18.jpg', '83.jpg', '17.jpg'], 'healthy_soybean': ['16.jpg', '13.jpg', '34.jpg']}
		]
		'''
		train_test = []
		path = self.get_path_out() + '/train'
		train_test.append(self.list_image_path(path))
		path = self.get_path_out() + '/test'
		train_test.append(self.list_image_path(path))
		return train_test

	
	def sampling(self):
		pass


class Split(Sampling):

	def __init__(self, path, percentage_test, percentage_validation):
		super().__init__(path)
		self.percentage_test = percentage_test
		self.percentage_validation =percentage_validation
		path_out = self.get_path() + '/train_test'
		self.set_path_out(path_out)

	def splitExamples(self, examples):
		'''
		examples: array com os exemplos de imagens
		Return
			[train, validation, test] ou
			[train, test] ou
		'''
		if self.percentage_validation > 0:
			test = int(self.percentage(self.percentage_test, len(examples)))
			validation = test + int(self.percentage(self.percentage_validation, len(examples)))
			split_list = np.split(examples, [test, validation])
			return (split_list[2], split_list[1], split_list[0])
		else:
			test = int(self.percentage(self.percentage_test, len(examples)))
			split_list = np.split(examples, [test])
			return (split_list[1], split_list[0])

	def percentage(self, percent, whole):
		return (percent * whole) / 100.0

	def define_split(self):
		'''
		retorna um dictionary onde a chave eh a classe e o valor eh uma lista de imagens divididas entre train, validation e test
		exemplo: 'soja':[[img1, img2][img3, img4][img5, img6, img7, img7]]
		'''
		images_dict = self.read_dataset()
		
		images_class = {}
		if self.percentage_validation > 0:
			for class_, images in images_dict.items():
				train, validation, test = self.splitExamples(images)
				tr_va_te = []
				tr_va_te.append(train)
				tr_va_te.append(test)					
				tr_va_te.append(validation)
				images_class[class_] = tr_va_te
		else:
			for class_, images in images_dict.items():
				train, test = self.splitExamples(images)
				tr_va_te = []
				tr_va_te.append(train)
				tr_va_te.append(test)
				images_class[class_] = tr_va_te

		return images_class


	def save_split(self, images_class):

		if self.percentage_validation > 0:
			directory_validation = self.path_out + '/' + 'validation/'
			self.createDirectory(directory_validation)

		directory_train = self.path_out + '/' + 'train/'		
		directory_test = self.path_out + '/' + 'test/'

		self.createDirectory(directory_test)
		
		for class_, list_train in images_class.items():
			for path in list_train[0]:								
				img = self.readImage(path)
				directory = directory_train + class_ + '/'
				self.createDirectory(directory)
				_, file = os.path.split(path)
				path_img = directory + file
				cv2.imwrite(path_img ,img)

			for path in list_train[1]:								
				img = self.readImage(path)
				directory = directory_test + class_ + '/'
				self.createDirectory(directory)
				_, file = os.path.split(path)
				path_img = directory + file
				cv2.imwrite(path_img ,img)

			if self.percentage_validation > 0:
				for path in list_train[2]:								
					img = self.readImage(path)
					directory = directory_validation + class_ + '/'
					self.createDirectory(directory)
					_, file = os.path.split(path)
					path_img = directory + file
					cv2.imwrite(path_img ,img)
			
	def sampling(self):

		path = self.get_path() + '/train_test'

		if not os.path.exists(path):
			print('Não existe train test, contruindo...')
			images_class = self.define_split()
			self.save_split(images_class)
		else:
			print('Treinamento e teste já existem.')

class Fold(Sampling):

	#folds/fold1/train/
	#folds/fold1/test/

	#folds/fold2/train/
	#folds/fold2/test/
	#...

	def __init__(self, path, number_fold, percentage_test, percentage_validation, _type):
		super().__init__(path)
		path_out = self.get_path() + '/folds'
		self.set_path_out(path_out)
		self.number_fold = number_fold
		self.percentage_test = percentage_test
		self.percentage_validation = percentage_validation
		self._type = _type # 0 -> k-fold, 1 -> Leave-One-Out 

	def defineFolds(self):
		'''
		separa as imagens em dobras para treinamento e teste
		'''
		cv = None
		if self._type == 0: 
			cv = ShuffleSplit(n_splits=self.number_fold, test_size=self.percentage_test, random_state=0)
			print("------------------------> ShuffleSplit ")
		elif self._type == 1:
			print("------------------------> LeaveOneOut ")
			cv = LeaveOneOut()
		
		images_dict = self.read_dataset()
		for class_, images in images_dict.items():
			cv.get_n_splits(images)
			number_fold = 1
			validation_index = None
			img_train, img_validation, img_test = None, None, None
			for train_index, test_index in cv.split(images):
				directory_train = self.path_out + '/fold' + str(number_fold) + '/' + 'train/' + class_ + '/'
				directory_test = self.path_out + '/fold' + str(number_fold) + '/' + 'test/' + class_ + '/'
				self.createDirectory(directory_train)
				self.createDirectory(directory_test)

				if self.percentage_validation > 0:
					directory_validation = self.path_out + '/fold' + str(number_fold) + '/' + 'validation/' + class_ + '/'
					self.createDirectory(directory_validation)
					percentage_validation = len(images) * (self.percentage_validation * 100) / 100
					validation_index = train_index[0:round(percentage_validation)]
					train_index = train_index[round(percentage_validation): len(train_index)]
					img_validation = [images[i] for i in validation_index]
					for path in img_validation:								
						img = self.readImage(path)
						_, file = os.path.split(path)
						path_img = directory_validation + file
						cv2.imwrite(path_img ,img)


				number_fold = number_fold + 1
				img_train = [images[i] for i in train_index] 
				img_test = [images[i] for i in test_index]
				for path in img_train:								
					img = self.readImage(path)
					_, file = os.path.split(path)
					path_img = directory_train + file
					cv2.imwrite(path_img ,img)

				for path in img_test:								
					img = self.readImage(path)
					_, file = os.path.split(path)
					path_img = directory_test + file
					cv2.imwrite(path_img ,img)
	
	def sampling(self):

		if not os.path.exists(self.get_path_out()):
			print('Não existe folds, contruindo folds...')
			self.defineFolds()
		else:
			print('Folds já existem.')

	
