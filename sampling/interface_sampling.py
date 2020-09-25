#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

if sys.version_info >= (3, 0):
	#from .sampling import ImageUtils
	from .sampling import Fold
	from .sampling import Split
	#from .sampling import ImageUtils
else:
	#from sampling import ImageUtils
	from sampling import Fold
	from sampling import Split
	#from sampling import ImageUtils


'''
Esta classe é uma interface do módulo de geração de amostragem.
Ela é responsável acionar outras classes para fazer a amostragem do banco de imagem
As amostragens pode ser validação cruzada, treino, validação e teste ou somente treino e validação.
'''

class SamplingMain:

	def __init__(self, sampling, percentage_train, percentage_validation, 
				 percentage_test, path_in, number_fold, _type):
		'''
		:param sampling: tipo de amostragem, 0 para validação cruzada, 1 para divisão em treinamento, validação e teste;
		:param percentage_train: valor de 0 a 100 para especificar o percentual de imagens usadas no treinamento
		:param percentage_validation: valor de 0 a 100 para especificar o percentual de imagens usadas na validação
		:param percentage_test: valor de 0 a 100 para especificar o percentual de imagens usadas no teste.
		:param path_in: diretório do banco de imagens
		:number_fold: número de folds da validação cruzada
		:percentage_test_fold: percentual de teste para cada fold
		Atenção: a soma dos valores percentage_train, percentage_validation, percentage_test deve ser no máximo 100 
		'''
		self.sampling = sampling
		self.percentage_train = percentage_train
		self.path_in = path_in
		self.sp = None
		if self.sampling == 0:				
			self.sp = Fold(path_in,number_fold, percentage_test, percentage_validation, _type)
		elif self.sampling == 1:
			self.sp = Split(path_in, percentage_test, percentage_validation)
		else:
			print ('Informe qual tipo de amostragem: 0 para validação cruzada ou 1 para treinamento, validação e teste')



		

	def images_train_test(self):
		return self.sp.images_train_test()
	
	def define_class(self):
		return self.sp.get_classes()

	'''
	#exemplo: './database/soybean_diseased_healthy/train_test' ou 
			  './database/soybean_diseased_healthy/folds/fold1'

	'''
	def get_path_out(self):
		return self.sp.get_path_out()

	#exemplo: './database/soybean_diseased_healthy'
	def get_path(self):
		return self.sp.get_path()

	def removeDirectory(self, directory):
		self.sp.removeDirectory(directory)

	def what_sampling(self):
		if self.sampling == 1:
			return 'Split'
		elif self.sampling == 0:
			return 'Cross'
		else:
			print ('No instance!')

	def run(self):
		#print(self.what_sampling())
		self.sp.sampling()


