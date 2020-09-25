#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

if sys.version_info >= (3, 0):
	from .extractors import Sift
	from .extractors import Surf
else:
	from extractors import Sift
	from extractors import Surf
	
import numpy as np
from os import listdir
from os.path import isfile, join


'''
Esta classe é uma interface do módulo de geração de sequências.
Ela é responsável por retornar sequências de strings extraídas de imagens. 
'''

class ExtractorsMain:

	def __init__(self, path, extractor, contrastThreshold):
		'''
		:param path: diretório onde estão as imagens para extrair as características.
		:param extractor: Qual extrator será usado. 0 - Sift, 1 - para Surf.
		'''
		self.path = path
		self.extractor = extractor
		self.contrastThreshold = contrastThreshold

	
	def run(self):
		'''
		:return: Um array com os descritores de cada imagem em formato vstack para ser usado pelo cluster.
		Um dicionário, onde a key é um id para a imagem e o value possui um array com duas posições, na primeira há
		os pontos de interesse da imagem, na segunda há os descritores da imagem.
		'''
		files = [f for f in listdir(self.path) if isfile(join(self.path, f))]

		ext = None
		if self.extractor == 0:
			ext = Sift(self.contrastThreshold)
		elif self.extractor == 1:
			ext = Surf(self.contrastThreshold)
		else:
			print ('Informe o extrator, 0 para Sift ou 1 para Surf.')

		
		images = {}
		cont = 0
		for f in files:
			file = self.path + '/' + f
			kp, ds = ext.features(file)
			if not isinstance(ds, np.ndarray): # caso não consiga key ponts da imagem
				print(file)
				cont = cont + 1
				continue
			if len(kp) < 2:
				print(file) 
				cont = cont + 1
				continue
			images[f] = [kp,ds]
		print('Imagens sem keypoints ->', cont)
			
		descriptor_vstack = np.array(range(128)) #apenas para iniciar o array

		for key, kp_ds in images.items():
			descriptor_vstack = np.vstack((descriptor_vstack, kp_ds[1]))
		
		return descriptor_vstack, images

'''
Example of the use
'''
if __name__ == '__main__':
	 ex = ExtractorsMain('path', 1)
	 print (ex.run())

