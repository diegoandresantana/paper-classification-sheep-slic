#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 
import numpy as np

class Extractor(object):
	def __init__(self, extractor):
		self.extractor = extractor
		self.name = None
		self.image = None

	def get_name(self):
		return self.name

	def set_name(self, name):
		self.name = name

	def features(self, image_gray):
		pass

	def gray(self, image):
		'''
		Convert image to gray image
		:param image: image to convert
		:return: gray image
		'''
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return gray

	def read_image(self, path):
		'''
		Read image of the path
		:param path: path to image
		:return: image
		'''
		img = cv2.imread(path)
		return img

	'''
	def remove_duplicate(self, keypoints, descriptors):

		kps = []
		for kp in keypoints:
			tup = (kp.pt[0],kp.pt[1])
			kps.append(tup)

		kps_unique = np.unique(kps, axis=0, return_index=True)
		tuples_xy = []
		for i in kps_unique:
			tuples_xy.append(tuple(i))

		indexes = []
		for i in tuples_xy[1]:
			indexes.append(i)

		kp = []
		ds = []
		for i in indexes:
			kp.append(keypoints[i])
			ds.append(descriptors[i])
		
		exit()

		return kp, ds
	'''
		


class Sift(Extractor):

	def __init__(self, contrast):
		super(Sift, self).__init__(cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=contrast, edgeThreshold=10, sigma=1.6))
		#super(Sift, self).__init__(cv2.SIFT())
		self.set_name('SIFT')

	def features(self, path_image):
		'''
		Read keypoints and feature of the gray image
		:param path_image: path image
		:return: keypoints extracted of the image
		:return: descriptors feature of each keypoints
		'''
		img = self.read_image(path_image)
		image_gray = self.gray(img)
		keypoints, descriptors = self.extractor.detectAndCompute(image_gray, None)
		#print(len(keypoints), len(descriptors))
		#keypoints, descriptors = self.remove_duplicate(keypoints, descriptors)
		#print(len(keypoints), len(descriptors))

		return [keypoints, descriptors]


class Surf(Extractor):

	def __init__(self, contrast):
		super(Surf, self).__init__(cv2.xfeatures2d.SURF_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=contrast, edgeThreshold=10, sigma=1.6))
		self.set_name('SURF')

	def features(self, path_image):
		'''
		Read keypoints and feature of the gray image
		:param image_gray: gray image
		:return: keypoints extracted of the image
		:return: descriptors feature of each keypoints
		'''
		img = self.read_image(path_image)
		image_gray = self.gray(img)
		keypoints, descriptors = self.extractor.detectAndCompute(image_gray, None)
		keypoints, descriptors = self.remove_duplicate(keypoints, descriptors)
		return [keypoints, descriptors]

'''
Example of the use
'''
if __name__ == '__main__':

	surf = Surf()
	sift = Sift()

	path = 'oidio_15.png'

	kp, ds = surf.features(path)
