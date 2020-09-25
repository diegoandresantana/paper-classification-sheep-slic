#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys

if sys.version_info >= (3, 0):
	from .clusters import VCKMeans
	from .clusters import VCHierarchical
else:
	from clusters import VCKMeans
	from clusters import VCHierarchical


import numpy as np

'''
Esta classe é uma interface do módulo de geração de sequências.
Ela é responsável por retornar sequências de strings extraídas de imagens. 
'''

class ClustersMain:

	def __init__(self, k, cluster, descriptor_vstack):
		'''
		:param k: tamanho do agrupamento.
		:param cluster: Qual cluster será usado. 0 - Kmeans, 1 - para AgglomerativeClustering.
		:param descriptor_vstack: vstack de dados para agrupamento
		'''
		self.k = k
		self.cluster = cluster
		self.descriptor_vstack = descriptor_vstack

	
	def run(self):
		'''
		:return: modelo do cluster
		'''
		
		cluster = None
		if self.cluster == 0:
			
			cluster = VCKMeans(self.k, self.descriptor_vstack)
		elif self.cluster == 1:
			cluster = VCHierarchical(self.k, self.descriptor_vstack)
		else:
			print ('Informe o extrator, 0 para KMeans ou 1 para Hierarchical.')

		return cluster.get_model()


