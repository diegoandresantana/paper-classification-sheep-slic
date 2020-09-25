#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance
from jgraph import *
import cv2 

class Vertice:

	def __init__(self, edge, edge_weight, vertice_label, xy, description,cluster_center):
		self.edge = edge
		self.edge_weight = edge_weight
		self.vertice_label = vertice_label 
		self.xy = xy # ponto no plano da vertice
		self.description = description
		self.cluster_center = cluster_center



'''
Esta classe em como objetivo representar uma imagem por meio de um grafo desconexo/conexo. Depois de representar
um grafo, cada componente é percorrida usando busca por profundidade ou largura. Uma string é extraída de uma
componente a medida que cada vertice é visitada pelo algoritmo de busca. Os labels que representam as vértices
são concatenados quando um vértice é visitado. Os algoritmos de buscas respeitam o peso de cada aresta para formar 
a string, ou seja, a que tem um peso menor (mais próxima) é visitada primeiro.
'''

class Forest:

	def __init__(self, vertices, type_search, minimum_size_sequence, path_img):
		self.vertices = vertices
		self.type_search = type_search
		self.minimum_size_sequence = minimum_size_sequence
		self.graph = Graph()
		self.path_img = path_img

	def create_graph(self):
		edges = []
		edges_weight = []
		edges_weight_label = []
		vertices_label = []
		xy = []
		descriptions = []
		cluster_centers = []
		for v in self.vertices:
			edges.append(v.edge)
			edges_weight.append(v.edge_weight)
			#print('******* ', v.edge_weight)
			edges_weight_label.append(str(round(v.edge_weight, 1)))
			#print('******* ', v.edge_weight)
			vertices_label.append(str(v.vertice_label))
			xy.append(v.xy)
			descriptions.append(v.description)
			cluster_centers.append(v.cluster_center)

		self.graph.add_vertices(len(self.vertices))
		self.graph.add_edges(edges)
		self.graph.es['weight'] = edges_weight
		self.graph.es['label'] = edges_weight_label
		self.graph.vs['label'] = vertices_label
		self.graph.vs['xy'] = xy
		self.graph.vs['description'] = descriptions
		self.graph.vs['cluster_center'] = cluster_centers


		#self.graph.simplify(loops=True, combine_edges=max) #  will take the maximum of the weights of multiple edges and assign that weight to the collapsed edge. 

	def join_components(self, components, weight_dxy):

		
		vertexes_central = []
		for component in components:
			vertex_central = self.identify_vertex_closer_center(component)
			vertexes_central.append(vertex_central)
		

		array_vertex = []
		edges = []
		edges_weight = []
		i = 0
		for vertex in vertexes_central:
			array_vertex.extend(vertexes_central[0:i]) # pega a primeira parte do vetor
			array_vertex.extend(vertexes_central[i+1:len(vertexes_central)]) #pega a segunda parte do vetor

			pt, pts, cluster_center, cluster_centers = self.create_points_to_measure(vertex, array_vertex)
			weight, index = self.closest(pts, pt, cluster_centers, cluster_center, weight_dxy)
			edge = (vertex.index,index)
			edges.append(edge)
			edges_weight.append(weight)
			i = i + 1	
			array_vertex = []


		#eds = self.graph.get_edgelist()
		#eds.extend(edges)
		edw = self.graph.es['weight']
		edw.extend(edges_weight)
		edl = self.graph.es['label']
		edl.extend([str(round(x, 1)) for x in edges_weight])

		self.graph.add_edges(edges)
		self.graph.es['weight'] = edw
		self.graph.es['label'] = edl

		#print(len(self.graph.get_edgelist()))
		#print(len(self.graph.es['weight']))
		#print(len(self.graph.es['label']))

		
		#exit()
		
		

	def create_points_to_measure(self, vertex, vertexes):

		#weight = self.graph.es[id_edge]['weight']
		pt_vertex = np.array([vertex.attributes()["xy"]])
		cluster_center = np.array([vertex.attributes()["cluster_center"]])
		pts = []
		cluster_centers = []
		for ver in vertexes:
			pts.extend(np.array([ver.attributes()["xy"]])) # pega somente o ponto do objeto			
			cluster_centers.extend(np.array([ver.attributes()["cluster_center"]])) # pega somente os centroides de cada objeto

		return pt_vertex, pts, cluster_center, cluster_centers

	def closest(self, points, point, cluster_centers, cluster_center, weight_dxy):

		
		disXY = distance.cdist(points, point,'euclidean').min(axis=1)
		dis_semantics = distance.cdist(cluster_centers, cluster_center,'euclidean').min(axis=1) # terao que ser ignorados os valores zeros
		
		ds = []
		for dxy, dis in zip(disXY, dis_semantics):
			wght = ((dxy * weight_dxy) + (dis * (1 - weight_dxy)))#calculo com peso
			ds.append(wght)

		index = ds.index(min(ds)) # pega o indice do ponto com menor valor (distancia)
		return ds[index], index #peso da aresta, indice do ponto mais proximo
		 

	def plot_graph(self):
		layout = self.graph.layout_fruchterman_reingold()
		'''
		print '29 label ----------->', self.graph.vs[29]['label']
		print '53 label ----------->', self.graph.vs[53]['label']
		print 'aresta 29 ----------->', self.graph.es[29].tuple
		print 'peso aresta 29----------->', self.graph.es[29]['weight']
		print '19 label ----------->', self.graph.vs[19]['label']
		print 'aresta 19 ----------->', self.graph.es[19].tuple
		print 'peso aresta 19----------->', self.graph.es[19]['weight']
		print 'peso aresta 53----------->', self.graph.es[53]['weight']
		'''		
		plot(self.graph, layout=layout)

	def get_point_center_image(self):
		height, width = cv2.cvtColor(cv2.imread(self.path_img), cv2.COLOR_BGR2GRAY).shape
		x = width/2
		y = height/2
		return np.array([(x,y)])

	def identify_vertex_closer_center(self, component):
		vertexes_of_the_component = self.graph.vs[component]
		points = []
		for vertex in vertexes_of_the_component:
			#print vertex['xy']
			points.extend(np.array([(vertex['xy'])])) # pega somente o ponto do objeto
		
		point = self.get_point_center_image()

		disXY = distance.cdist(points, point,'euclidean').min(axis=1)
		index = disXY.tolist().index(min(disXY)) # pega o indice do ponto com menor valor (distancia)
		#print('------->',index)
		return vertexes_of_the_component[index]

	def sort_by_weight(self, neighbors, v):
		id_edges = []
		for n in neighbors: # para todos os vizinhos
			id_edges.append( self.graph.get_eid(v,n)) # recupera o id da aresta da vertice em questao 

		
		vertexes = []
		for vertex, id_edge in zip(neighbors,id_edges): # percorre todas as vertices vizinhas
			weight = self.graph.es[id_edge]['weight'] # pega o peso da aresta
			vertexes.append((vertex,weight)) # cria uma tupla com o id da vertice e o peso exemplo (10, 3.40)

		#print(vertexes)
		vertexes.sort(key=lambda tup: tup[1]) # ordena a lista de tuplas pelo peso

		vertexes_sorted = []
		for vertex in vertexes: # percorre todas as vertices ordenadas para pegar somente o is das vertices
			vertexes_sorted.append(vertex[0]) # cria um array com os ids das vertices em ordem, ou seja, ordenas os vizinhos
		return vertexes_sorted # retorna os vizinhos ordenados pelo peso.

	# Breadth-First Search
	def breadth_first_search(self, component, start): 
		#print component, start
		graph = {}
		for vertex in component: # lista todas as vertices da componente
			neighbors = self.graph.neighbors(vertex, mode=ALL) # pega todas as vertices adjacentes de uma determinada vertice			
			graph[vertex] = self.sort_by_weight(neighbors, vertex) # ordena as vertices adjacentes por peso e adiciona o item ao dicionario que representa um grafo
		# keep track of all visited nodes
		#print 'largura ------> ', graph
		explored = []
		# keep track of nodes to be checked
		queue = [start]
		
		# keep looping until there are nodes still to be checked
		while queue:
			# pop shallowest node (first node) from queue
			node = queue.pop(0)
			if node not in explored:
				# add node to list of checked nodes
				explored.append(node)
				neighbours = graph[node]
	 
				# add neighbours of node to queue
				for neighbour in neighbours:
					queue.append(neighbour)		
		return explored

	# Deep-First Search
	def deep_first_search(self, component, start): 
		#print component, start
		graph = {}
		for vertex in component: # lista todas as vertices da componente
			neighbors = self.graph.neighbors(vertex, mode=ALL) # pega todas as vertices adjacentes de uma determinada vertice			
			ordened = self.sort_by_weight(neighbors, vertex) # ordena as vertices adjacentes por peso e adiciona o item ao dicionario que representa um grafo 
			# inverte a ordem devido a busca ser por profundidade. Assim a prioridade fica para o mais proximo
			graph[vertex] = ordened[::-1]
		# keep track of all visited nodes
		explored = []
		# keep track of nodes to be checked
		stack = [start]
		# keep looping until there are nodes still to be checked
		while stack:
			# pop node (last node) from stack
			node = stack.pop()
			if node not in explored:
				# add node to list of checked nodes
				explored.append(node)
				neighbours = graph[node]
				# add neighbours of node to queue
				for neighbour in neighbours:
					stack.append(neighbour)
		return explored
		
	def search(self, component, start):
		sequence_vertex_ids_visited = None
		if self.type_search == 0:
			sequence_vertex_ids_visited = self.breadth_first_search(component, start)			
		else:
			sequence_vertex_ids_visited = self.deep_first_search(component, start)
		# usando os ids das vertices pega somente os labels das vertices, ou seja, os labels que representa o alfabeto 
		sequence_labels = self.graph.vs[sequence_vertex_ids_visited]['label'] 
		#return ''.join(sequence_labels)
		return sequence_labels

	def get_large_string(self, strings):
		size = 0
		index = 0
		i = 0
		for string in strings:
			if len(string) > size:
				size = len(string)
				index = i
			i = i + 1
		return index

	def define_string(self, weight_dxy):
		strings = []
	
		components = self.graph.components()
		#print('antes -> ',len(components))
		if self.minimum_size_sequence == -1:			

			if len(components) > 1:
				len_component = len(components)
				while len_component != 1:
					self.join_components(components, weight_dxy)
					components = self.graph.components()
					len_component = len(components)
				
		
		#print('depois -> ',len(components))	
		for component in components:
			if len(component) >= self.minimum_size_sequence:
				vertex = self.identify_vertex_closer_center(component) # identifica a vertice da componente mais proxima do ponto 0,0 no plano
				string = self.search(component, vertex.index)
				strings.append(string)
		# pega somente a maior string do grafo, ou seja, cada image contribui apenas com uma string
		if self.minimum_size_sequence == 0:
			index = self.get_large_string(strings)
			strings = [strings[index]]	

		return strings	
