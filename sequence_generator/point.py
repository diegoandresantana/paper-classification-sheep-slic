#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

if sys.version_info >= (3, 0):
	from .graph import Vertice
else:
	from graph import Vertice


import numpy as np
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances_argmin_min
import cv2 




class KeyPoint:

	def __init__(self):
		self.xy = None # posicao x y do ponto no plano da imagem
		self.description = None # descricao do ponto, fornecida pelo descritor de caracteristica
		self.label = None # label, do alfabeto, que representa o ponto
		self.cluster_center = None # cluster que o ponto (self) em questao pertence
		self.closest = None # ponto mais proximo do ponto (self) em questao
		self.weight_edge = None # peso da aresta
		self.representation_vertice = None # numero da vertice que representa o ponto no grafo
		self.representation_vertice_closest = None #numero da vertice que representa o ponto mais proximo no grafo.
				

class KeyPointUtils:
	def __init__(self):
		self.keyPoints = [] # list of the keypoints
		self.cont = 0
		self.cont1 = 0
	
	def create_vertices(self):
		vertices = []

		for kp in self.keyPoints:
			edge = (kp.representation_vertice, kp.representation_vertice_closest)
			edge_weight = kp.weight_edge
			vertice_label = kp.label
			xy = kp.xy
			description = kp.description
			cluster_center = kp.cluster_center
			v = Vertice(edge, edge_weight, vertice_label, xy,description,cluster_center)
			vertices.append(v)
		return vertices


	def closest(self, points, point, cluster_centers, cluster_center, weight_dxy):

		
		disXY = distance.cdist(points, point,'euclidean').min(axis=1)
		dis_semantics = distance.cdist(cluster_centers, cluster_center,'euclidean').min(axis=1) # terao que ser ignorados os valores zeros
		
		ds = []
		for dxy, dis in zip(disXY, dis_semantics):
			#devo percorrer e calcular a distancia com os pesos e ir adicionando em um outro vetor (ds)
			#quando tiver zero em dis_semantics devo colocar um valor muito alto no vetor (ds), isso devido o valor 0 ser do mesmo cluster do ponto base
			#no fim pego o menor valor do vetor (ds)
			'''
			if dis == 0:
				ds.append(100000)
				continue
			'''
			wght = ((dxy * weight_dxy) + (dis * (1 - weight_dxy)))#calculo com peso
			ds.append(wght)

		index = ds.index(min(ds)) # pega o indice do ponto com menor valor (distancia)
		return points[index], ds[index], index #retorna o ponto mais proximo, peso da aresta, indice do ponto mais proximo
		
	
	def create_points_to_measure(self, point, points):
		pt = np.array([(point.xy)])
		#print pt
		cluster_center = np.array([(point.cluster_center)])
		pts = []
		cluster_centers = []
		for p in points:
			pts.extend(np.array([(p.xy)])) # pega somente o ponto do objeto			
			cluster_centers.extend(np.array([(p.cluster_center)])) # pega somente os centroides de cada objeto

		return pt, pts, cluster_center, cluster_centers





	def get_point_center_image(self, path_img):
		points = []
		for p in self.keyPoints:
			points.extend(np.array([(p.xy)])) # pega somente o ponto do objeto	

		height, width = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2GRAY).shape
		x = width/2
		y = height/2
		point = np.array([(x,y)]) # cria o ponto do centro da imagem
		disXY = distance.cdist(points, point,'euclidean').min(axis=1)
		index = disXY.tolist().index(min(disXY)) # pega o indice do ponto com menor valor (distancia)
		return index # retorna o index do ponto mais próximo do centro

	def closest_from_center(self, points, point):		
		disXY = distance.cdist(points, point,'euclidean').min(axis=1)
		index = disXY.tolist().index(min(disXY))  # pega o indice do ponto com menor valor (distancia)
		return index #retorna o indice do ponto mais proximo


	def create_representation_graph_from_center(self, path_img):
		#print('passou',1)
		index_center_point = self.get_point_center_image(path_img)		
		others_points = []
		others_points.extend(self.keyPoints[0:index_center_point])
		others_points.extend(self.keyPoints[index_center_point+1:len(self.keyPoints)]) 
		point_center = self.keyPoints.pop(index_center_point)

		
		ordened_points = []
		ordened_points.append(point_center) #coloca o ponto central na primiera posição
		bucket = len(ordened_points)
		while others_points and bucket > 0:
			pt, pts, cluster_center, cluster_centers = self.create_points_to_measure(point_center, others_points)
			index_closest = self.closest_from_center(pts, pt) # pega o ídice do ponto mais próximo
			point_closest = others_points.pop(index_closest) # remove o ponto mais pŕoximo da lista
			#point_closest.representation_vertice_closest = index_closest # representa o ponto mais próximo com o índice
			ordened_points.append(point_closest) # adiciona os pontos mais próximo em uma lista, dessa forma eles serão ordenados

		i = 0
		for kp in ordened_points:
			if i < (len(ordened_points) - 1):
				kp.closest = ordened_points[i+1]
				kp.weight_edge = 1
				kp.representation_vertice = i
				kp.representation_vertice_closest = i+1
			else: # última vértice
				kp.closest = ordened_points[i]
				kp.weight_edge = 1
				kp.representation_vertice = i
				kp.representation_vertice_closest = i
			i = i + 1

		self.keyPoints = ordened_points # devolve os pontos com a representação gráfica para a lista

	

	def create_representation_graph(self, weight_dxy):
		i = 0
		arr = [] # usado para criar um vetor com os pontos, exceto o ponto base usado para identificar o mais proximo dele
		for kp in self.keyPoints:
			arr.extend(self.keyPoints[0:i]) # pega a primeira parte do vetor
			arr.extend(self.keyPoints[i+1:len(self.keyPoints)]) #pega a segunda parte do vetor
			#point =  np.array([(self.keyPoints[i][0], self.keyPoints[i][1])]) # pega o ponto base
			# arr -> passa o array de pontos menos o base. point -> ponto base
			pt, pts, cluster_center, cluster_centers = self.create_points_to_measure(kp, arr)
			closest, weight_edge, closest_name = self.closest(pts, pt, cluster_centers, cluster_center, weight_dxy)
			kp.closest = closest
			kp.weight_edge = weight_edge
			kp.representation_vertice = i
			kp.representation_vertice_closest = closest_name 

			i = i + 1	
			arr = []

	def get_cluster_centers(self, label, model_kmeans):	
		'''
		Get cluster center to the label
		:param label: label of the cluster
		:param model_kmeans: cluster model
		:return: cluster center to the label
		'''
		return model_kmeans.cluster_centers_[int(label)]

	def define_cluster_to_point(self, model, new_point):	
		'''
		Define the cluster of the new point
		:param new_point: new point
		:param model: cluster model
		:return: cluster of the point
		'''
		
		return model.predict(new_point)[0]

	
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

		if weight_dxy != 0:
			self.create_representation_graph(weight_dxy) #cria um grafo baseado em distância
		else:
			self.create_representation_graph_from_center(path_img) #cria um caminho começando pelo centro.
		return self.create_vertices()
		

