from numpy import array
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional

from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

csv_logger = CSVLogger('log.csv', append=True, separator=';')

from keras.models import load_model
import random

class Preprocessing():
    
    def __init__(self,input_file):
        self.input_data_file = input_file
        self.data = None
        self.vocab_size = None
        self.encoded_data = None
        self.max_length = None
        self.sequences = None
        self.x = None
        self.y = None
        self.tokenizer = None
    
    def load_data(self):
        fp = open(self.input_data_file,'r')
        self.data = fp.read().splitlines()        
        fp.close()
        
    def encode_data(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.data)
        self.encoded_data = self.tokenizer.texts_to_sequences(self.data)
        print(self.encoded_data)
        self.vocab_size = len(self.tokenizer.word_counts)+1
        
    def generate_sequence(self):
        seq_list = list()
        for item in self.encoded_data:
            l = len(item)
            for id in range(1,l):
                seq_list.append(item[:id+1])
        self.max_length = max([len(seq) for seq in seq_list])
        self.sequences = pad_sequences(seq_list, maxlen=self.max_length, padding='pre')
        print(self.sequences)
        self.sequences = array(self.sequences)
            
    def get_data(self):
        self.x = self.sequences[:,:-1]
        self.y = self.sequences[:,-1]
        print("y before:",self.y)
        self.y = to_categorical(self.y,num_classes=self.vocab_size)
        print("y After:",self.y)

pr = Preprocessing('/content/gdrive/My Drive/3_Semester_Analyse_Projekt/LSTM/EmbeddingLayer/EmbeddingBidirectionalLSTM/smallBatchSize/speeches-afd-small.txt')
pr.load_data()
pr.encode_data()
pr.generate_sequence()
pr.get_data()


#model = load_model("lang_model.h5")
model = load_model(".mdl_wts.hdf5")


#earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
#mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
#mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='loss', mode='min')
#earlyStopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='loss', mode='auto')
#reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='auto')
#self.history = self.model.fit(self.x,self.y,epochs=self.epochs, callbacks=[csv_logger, mcp_save])
history = model.fit(pr.x,pr.y,epochs=50, callbacks=[csv_logger, mcp_save ], batch_size=128)
#self.history = self.model.fit(self.x,self.y,epochs=self.epochs, callbacks=[csv_logger, reduce_lr_loss, mcp_save, earlyStopping ])
#self.history = self.model.fit(self.x,self.y,epochs=self.epochs, callbacks=[csv_logger, mcp_save, earlyStopping ])


model.save("lang_model.h5")