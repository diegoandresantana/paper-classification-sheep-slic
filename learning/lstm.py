#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 26 de jan de 2018

@author: Gilberto Astolfi
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics


class LSTMSyntactic:
   
    def __init__(self, k, x_train, y_train, x_test, y_test, nb_epoch, lstm_memory,path_models_checkpoints):
        
        self.k = k
        self.x_train = x_train 
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # fix random seed for reproducibility
        np.random.seed(7)
        self.nb_epoch = nb_epoch
        self.top_words = None
        # truncate and pad input sequences
        self.max_review_length = None
        self.embedding_vecor_length = 32
        self.model = None
        self.lstm_memory = lstm_memory
        self.path_models_checkpoints = path_models_checkpoints
       
    def __define_class(self,values):
        output = []
        seen = set()
        for value in values:
            if value not in seen:
                output.append(value)
                seen.add(value)
        return output
    
    def pad_sequence(self, vocabulary_size):
        tam = []
        for seq in self.x_train:
            tam.append(len(seq))

        #print(tam)
        #print('media -> ', np.mean(tam))
        #print('min ->', np.amin(tam,0))
        #print('max ->', np.amax(tam,0))
        #exit()

        self.x_train = sequence.pad_sequences(self.x_train, maxlen=vocabulary_size)
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=vocabulary_size)

        
        #exit()
        #print (len(self.x_train[0]))
        #print (len(self.x_train[1]))
        self.top_words = self.k + 1
        # truncate and pad input sequences
        self.max_review_length = len(self.x_train[0])
    
   
    def trainModel(self):
        """
        uses keras LSTM classifier 
        """
        print('----------------------------->',self.max_review_length)

        self.model = Sequential()
       
        self.model.add(Embedding(self.top_words, self.embedding_vecor_length, input_length=self.max_review_length))
        self.model.add(LSTM(self.lstm_memory, dropout=0.2, recurrent_dropout=0.2))
        # get number of class
        classes = self.__define_class(self.y_train)

        #print('x_train',len(self.x_train))
        #print('y_train',len(self.y_train))
       # print('x_test',len(self.x_test))
       # print('y_test',len(self.y_test))
       # print('len classes', len(classes))
       # print('classes', classes)

        
        if len(classes) > 2:      
            self.model.add(Dense(len(classes), activation='softmax'))         
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
        else:
            # to only two class, binary problem
            self.model.add(Dense(1, activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
       
        checkpoint = ModelCheckpoint(self.path_models_checkpoints, monitor='val_acc',
                                 verbose=1, save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)


        
        self.model.fit(self.x_train, self.y_train, epochs=self.nb_epoch, batch_size=64, 
                  callbacks=[checkpoint], validation_data=(self.x_test, self.y_test))

         
    
    def testModel(self):
        self.model.load_weights(self.path_models_checkpoints)
        predictions = self.model.predict_classes(self.x_test)
        classes = self.y_test        
        
        return (predictions, np.asarray([int(i) for i in classes]))      
        
