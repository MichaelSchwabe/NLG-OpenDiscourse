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

class Prediction():
    def __init__(self,tokenizer,max_len):
        self.model = None
        self.tokenizer = tokenizer
        self.idx2word = {v:k for k,v in self.tokenizer.word_index.items()}
        self.max_length = max_len
    
    def load_model(self):
        self.model = load_model("lang_model.h5")
        
    def predict_sequnce(self,text,num_words):
        for id in range(num_words):
            encoded_data = self.tokenizer.texts_to_sequences([text])[0]
            padded_data = pad_sequences([encoded_data],maxlen = self.max_length-1,padding='pre')
            y_pred = self.model.predict(padded_data)
            y_pred = np.argmax(y_pred)
            predict_word = self.idx2word[y_pred]
            text += ' ' + predict_word
        return text



pred = Prediction(pr.tokenizer,pr.max_length)    
pred.load_model()

min=1
max=5
rangeperseed=50

textlist=[]
print("####### SEED: Liebe Kollegen und Kolleginen ##########")
textlist.append("####### SEED: Liebe Kollegen und Kolleginen ##########")
for i in range(rangeperseed):
    if i is 0:
        text = pred.predict_sequnce("Liebe Kollegen und Kolleginen",random.randint(min, max))
        #print(text)
    elif i == rangeperseed-1:
        text = pred.predict_sequnce(text,random.randint(min, max))
        print(text)
        textlist.append(text)
    else:
        text = pred.predict_sequnce(text,random.randint(min, max))
        #print(text)
    


print("####### SEED: Die Energiewende ##########")
textlist.append("####### SEED: Die Energiewende ##########")
for i in range(rangeperseed):
    if i is 0:
        text = pred.predict_sequnce("Die Energiewende",random.randint(min, max))
        #print(text)
    elif i == rangeperseed-1:
        text = pred.predict_sequnce(text,random.randint(min, max))
        print(text)
        textlist.append(text)
    else:
        text = pred.predict_sequnce(text,random.randint(min, max))
        #print(text)
    
    

print("####### SEED: Die Legislaturperiode ##########")
textlist.append("####### SEED: Die Legislaturperiode ##########")
for i in range(rangeperseed):
    if i is 0:
        text = pred.predict_sequnce("Die Legislaturperiode",random.randint(min, max))
        #print(text)
    elif i == rangeperseed-1:
        text = pred.predict_sequnce(text,random.randint(min, max))
        print(text)
        textlist.append(text)
    else:
        text = pred.predict_sequnce(text,random.randint(min, max))
        #print(text)
   


print("####### SEED: Die Bundeswehr ##########")
textlist.append("####### SEED: Die Bundeswehr ##########")
for i in range(rangeperseed):
    if i is 0:
        text = pred.predict_sequnce("Die Bundeswehr",random.randint(min, max))
        #print(text)
    elif i == rangeperseed-1:
        text = pred.predict_sequnce(text,random.randint(min, max))
        print(text)
        textlist.append(text)
    else:
        text = pred.predict_sequnce(text,random.randint(min, max))
        #print(text)
    

print("####### SEED: NONE ##########")
textlist.append("####### SEED: NONE ##########")
for i in range(rangeperseed):
    if i is 0:
        text = pred.predict_sequnce("",random.randint(min, max))
        #print(text)
    elif i == rangeperseed-1:
        text = pred.predict_sequnce(text,random.randint(min, max))
        print(text)
        textlist.append(text)
    else:
        text = pred.predict_sequnce(text,random.randint(min, max))
        #print(text)
     
print("####### SEED: NONE ##########")
textlist.append("####### SEED: NONE ##########")
for i in range(rangeperseed):
    if i is 0:
        text = pred.predict_sequnce("",random.randint(min, max))
        #print(text)
    elif i == rangeperseed-1:
        text = pred.predict_sequnce(text,random.randint(min, max))
        print(text)
        textlist.append(text)
    else:
        text = pred.predict_sequnce(text,random.randint(min, max))
        #print(text) 

print("####### SEED: Wie wirkt sich die Energiewende auf die Bundesrepublik aus? ##########")
textlist.append("####### SEED: NONE ##########")
for i in range(rangeperseed):
    if i is 0:
        text = pred.predict_sequnce("Wie wirkt sich die Energiewende auf die Bundesrepublik aus?",random.randint(min, max))
        #print(text)
    elif i == rangeperseed-1:
        text = pred.predict_sequnce(text,random.randint(min, max))
        print(text)
        textlist.append(text)
    else:
        text = pred.predict_sequnce(text,random.randint(min, max))
        #print(text) 


with open('generated.txt', 'w') as f:
    for item in textlist:
        f.write("%s\n" % item)