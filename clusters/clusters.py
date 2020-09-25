# clustering dataset
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min



class Cluster(object):

	def __init__(self, k, model):
		self.name = None
		self.model = model
		self.k = k

	def get_name(self):
		return self.name

	def set_name(self, name):
		self.name = name

	def get_model(self):
		return self.model

	def define_cluster_to_point(self, new_point):	
		'''
		Define the cluster of the new point
		:param new_point: new point
		:return: cluster of the point
		'''
		pass

class VCKMeans(Cluster):

	def __init__(self, k, data):
		super(VCKMeans, self).__init__(k, KMeans(n_clusters=k, random_state=0).fit(data))
		self.set_name('KMeans')

	def define_cluster_to_point(self, new_point):	
		'''
		Define the cluster of the new point
		:param new_point: new point
		:return: cluster of the point
		'''
		# calculate distances between centroids and new point 
		_, distances = pairwise_distances_argmin_min(self.model.cluster_centers_, new_point)
		#print '->', distances
		# get the index of the small distance
		small_distance = np.argmin(distances)
		return small_distance


class VCHierarchical(Cluster):

	def __init__(self, k, data):
		super(VCHierarchical, self).__init__(k, AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(data))
		self.set_name('Hierarchical')



'''
Example of the use
'''
if __name__ == '__main__':

	X = np.array([
	[ 41.,  25.,   2.,   3.,  11.,  10.],
	[  11.,   10.,  0., 10.,   1.,  11.],
	[  0.,   0.,   1.,   0.,   0.,  11.],
	[  8.,   3.,  14.,  12.,  17.,  11.],
	[  3.,  11.,   6.,   1.,   3.,  11.],
	[  5.,  10.,   6.,   0.,   0.,  11.],
	[  1.,  10.,   6.,   0.,   3.,  11.],
	[  4.,  12.,   51.,   4.,   0.,  11.],
	[ 18.,   9.,  25.,   8.,   6.,  11.]])

	k = 3
	new_data = np.array([[ 4.,  12.,   51.,   4.,   0.,  11.]])
	
	print ('KMeans')
	km = VCKMeans(k, X)	
	cluster_label = km.define_cluster_to_point(new_data)
	print (cluster_label)
	print (km.get_model().labels_)

	'''
	print 'Hierarchical'
	hi = VCHierarchical(k, X)
	cluster_label = hi.define_cluster_to_point(new_data)
	print cluster_label
	print km.get_model().labels_
	'''

	
	

