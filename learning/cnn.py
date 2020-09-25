from __future__ import print_function#, unicode_literals
import time
import os
import tensorflow
import shutil
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from tensorflow import device
#from keras.applications import Xception, VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, DenseNet201, NASNetLarge, NASNetMobile
from keras.applications import Xception, ResNet50, Xception, VGG16, VGG19, ResNet50, InceptionV3, MobileNet, InceptionResNetV2 , DenseNet201, NASNetLarge, NASNetMobile,  MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, load_model
from keras import backend
from keras.backend import clear_session
from keras.utils import multi_gpu_model
from keras import layers
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#import pandas as pd
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress warnings


IMG_WIDTH, IMG_HEIGHT = 256, 256
SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
architectures = {"Xception":Xception, "VGG16":VGG16, "VGG19":VGG19, "ResNet50":ResNet50, "InceptionV3":InceptionV3,"InceptionResNetV2":InceptionResNetV2, "DenseNet201":DenseNet201, "NASNetLarge": NASNetLarge, "NASNetMobile":NASNetMobile,"MobileNetV2":MobileNetV2}

#architectures = {"Xception":Xception, "ResNet50":ResNet50}


class CNN:
	def __init__(self, train_dir, validation_dir, test_dir, batch_size, architecture, epochs,fineTuningRate, 
		path_models_checkpoints):

		if architecture == 'NASNetMobile':
			IMG_WIDTH = 331
			IMG_HEIGHT = 331
		
		self.train_dir = train_dir
		self.validation_dir = validation_dir
		self.test_dir = test_dir
		self.batch_size = batch_size
		self.architecture = architecture
		self.weights = 'imagenet'
		self.epochs = epochs
		self.fineTuningRate = fineTuningRate
		self.path_models_checkpoints = path_models_checkpoints
		self.learning_rate = 0.0001
		self.model = None
		self.history_validation = None
		self.history_test = None
		self.history_train = None

	def get_history_validation(self):
		return self.history_validation
		
	def set_history_validation(self, history_validation):
		self.history_validation = history_validation


	def get_history_test(self):
		return self.history_test
		
	def set_history_test(self, history_test):
		self.history_test = history_test

	def get_history_train(self):
		return self.history_train
		
	def set_history_train(self, history_train):
		self.history_train = history_train

	def get_learning_rate(self):
		return self.learning_rate
		
	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate

	def get_path_models_checkpoints(self):
		return self.path_models_checkpoints
		
	def set_path_models_checkpoints(self, path_models_checkpoints):
		self.path_models_checkpoints = path_models_checkpoints

	def get_fineTuningRate(self):
		return self.fineTuningRate
		
	def set_fineTuningRate(self, fineTuningRate):
		self.fineTuningRate = fineTuningRate

	def get_epochs(self):
		return self.epochs
		
	def set_epochs(self, epochs):
		self.epochs = epochs

	def get_weights(self):
		return self.architecture
		
	def set_weights(self, weights):
		self.weights = weights

	def get_architecture(self):
		return self.architecture
		
	def set_architecture(self, architecture):
		self.architecture = architecture

	def get_batch_size(self):
		return self.batch_size
		
	def set_batch_size(self, batch_size):
		self.batch_size = batch_size

	def get_train_dir(self):
		return self.train_dir
		
	def set_train_dir(self, train_dir):
		self.train_dir = train_dir

	def get_validation_dir(self):
		return self.validation_dir
		
	def set_validation_dir(self, validation_dir):
		self.validation_dir = validation_dir

	def get_test_dir(self):
		return self.test_dir
		
	def set_test_dir(self, test_dir):
		self.test_dir = test_dir

	def get_model(self):
		return self.model
		
	def set_model(self, model):
		self.model = model


	def get_train_generator(self):

		train_datagen = ImageDataGenerator( rescale=1. / 255, horizontal_flip=True, fill_mode="nearest", zoom_range=0.3, 
			width_shift_range=0.3, height_shift_range=0.3, rotation_range=30)

		train_generator = train_datagen.flow_from_directory(self.get_train_dir(), target_size=(IMG_HEIGHT, IMG_WIDTH), 
			batch_size=self.get_batch_size(), shuffle=True, class_mode="categorical")
		return train_generator


	def get_validation_generator(self):


		if self.get_validation_dir() == None:
			return None

		validation_datagen = ImageDataGenerator(rescale=1. / 255)
		validation_generator = validation_datagen.flow_from_directory(self.get_validation_dir(), target_size=(IMG_HEIGHT, 
			IMG_WIDTH), batch_size=self.get_batch_size(), shuffle=True, class_mode="categorical")
		return validation_generator

	def get_test_generator(self):
		
		test_datagen = ImageDataGenerator(rescale=1. / 255)
		test_generator = test_datagen.flow_from_directory(self.get_test_dir(), batch_size=1, shuffle=False, 
			target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode="categorical")
		return test_generator


	def create_model(self, architecture):
		model = architecture(weights='imagenet', include_top=False, input_shape=SHAPE)
		return model

	def trainModel(self):
		
		#with device('/cpu:0'):
		model = self.create_model(architecture = architectures[self.get_architecture()])

		if self.get_fineTuningRate() != -1:
		# calculate how much layers won't be retrained according on fineTuningRate parameter
			n_layers = len(model.layers)
			last_layers = n_layers - int(n_layers * (self.get_fineTuningRate() / 100.))
			for layer in model.layers[:last_layers]:
				layer.trainable = False

		else:  # without transfer learning
			self.set_weights(None)
			for layer in model.layers:
				layer.trainable = True

		
		# Adding custom Layers
		new_custom_layers = model.output
		new_custom_layers = Flatten()(new_custom_layers)
		new_custom_layers = Dense(1024, activation="relu")(new_custom_layers)
		new_custom_layers = Dropout(0.5)(new_custom_layers)
		new_custom_layers = Dense(1024, activation="relu")(new_custom_layers)
		try:
			num_classes = self.get_train_generator().num_class
		except:
			num_classes = self.get_train_generator().num_classes
		
		predictions = Dense(num_classes, activation="softmax")(new_custom_layers)

		# creating the final model
		model_final = Model(inputs=model.input, outputs=predictions)

		# compile the model
		try:
			model_final = multi_gpu_model(model_final, gpus=2)#, cpu_merge=1) #
			#tensorflow.device_scope('/gpu:0')
		except:
			pass
		#model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.Adagrad(), metrics=["accuracy"])
		model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=self.get_learning_rate(), momentum=0.9),
							metrics=["acc"])


		# Save the model according to the conditions
		if self.get_validation_generator() != None:
			checkpoint = ModelCheckpoint(self.get_path_models_checkpoints(), monitor='val_acc',
									 verbose=1, save_best_only=True, save_weights_only=False,
									 mode='auto', period=1)
		else:
			checkpoint = ModelCheckpoint(self.get_path_models_checkpoints(), monitor='acc',
									 verbose=1, save_best_only=True, save_weights_only=False,
									 mode='auto', period=1)


		#print(model_final.summary())

		# Train the model with validation
		if self.get_validation_generator() != None:
			history = model_final.fit_generator(self.get_train_generator(), 
				steps_per_epoch=self.get_train_generator().samples // self.get_batch_size(), 
				epochs=self.get_epochs(), validation_data=self.get_validation_generator(), 
				validation_steps=self.get_validation_generator().samples // self.get_batch_size(), 
				callbacks=[checkpoint])
			self.set_history_validation(history)
		else:
			history = model_final.fit_generator(self.get_train_generator(), 
				steps_per_epoch=self.get_train_generator().samples // self.get_batch_size(), 
				epochs=self.get_epochs(), callbacks=[checkpoint])
			self.set_history_train(history)


		#plot(H.history, file_name)
		#print ("Total time to train: %s" % (get_time(time.time() - START_TIME)))
		model_final = load_model(self.get_path_models_checkpoints())
		self.set_model(model_final)


	def testModel(self):

		test_features = self.get_model().predict_generator(self.get_test_generator(), self.get_test_generator().samples, verbose=1)
		y_pred = np.argmax(test_features, axis=1)
		y_true = self.get_test_generator().classes
		clear_session()
		return y_pred, y_true



		
