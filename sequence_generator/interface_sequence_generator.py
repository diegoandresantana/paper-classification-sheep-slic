#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

if sys.version_info >= (3, 0):
	from .point import KeyPointUtils
	from .graph import Forest
else:
	from point import KeyPointUtils
	from graph import Forest


import numpy as np


'''
Esta classe é uma interface do módulo de geração de sequências.
Ela é responsável por retornar sequências de strings extraídas de imagens. 
'''

class SequenceGeneratorMain:

	def __init__(self, cluster_model, search_type, weight_xy, minimum_size_sequence):
		'''
		:param cluster_model: o cluster conseguido com os descritores de um conjunto de imagens.
		:param search_type: tipo de busca no grafo para gerar as sequências. 0 - Breadth-First Search ou 1 - Deep-First Search.
		:param weight_xy: valor float, variando de 0 a 1, que especifica o peso dado a distância espacial dos pontos na geração do grafo de proximidade
		:param minimum_size_sequence: tamanho mínimo da sequência a ser extraída da imagem.
		'''
		self.cluster_model = cluster_model
		self.search_type = search_type
		self.weight_xy = weight_xy
		self.minimum_size_sequence = minimum_size_sequence

	def run(self, keypoint, descriptor, path_img):
		'''
		:param keypoint: um array com os pontos de interesses extraídos de uma imagem.
		:param descriptor: um array com a descrição de cada ponto de interesse da imagem.
		:return: um array, onde cada posição representa um outro array contendo uma sequência de caracteres extraídas de uma imagem
		exemplo de retorno [['1','3','4'], ['3','3','2']]
		'''
		kpy = KeyPointUtils()
		vertices = kpy.create_graph(keypoint, descriptor, self.cluster_model, self.weight_xy, path_img)
		fo = Forest(vertices, self.search_type, self.minimum_size_sequence, path_img)
		fo.create_graph()
		#fo.plot_graph()
		#exit()
		return fo.define_string(self.weight_xy)
