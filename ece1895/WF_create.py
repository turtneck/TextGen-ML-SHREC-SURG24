from fun_colors import *
import os,shutil
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop


global dir_path
dir_path = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__": print(f"DIRECTORY:\t\t{dir_path}>")

#helpers
def red(filename):
    arr=[]
    with open(dir_path+"/"+filename, 'r') as f:
        line = f.readline()
        for i in line.split(" "):
            if i not in [" ","\n"]:
                arr.append(i)
    return arr

#check that the dictionary is stable (in order)
#def dict_review():
    


#num_words: number of words to look at previous from the word it's predicting
#unique_tokens:number of words in dictionary

    
class WF_model:
    #first init
    def __init__(self,name,num_words=10,verbose=False):
        self.verbose=verbose
        self.num_words=num_words
        self.dict_size=0
        open('dict.txt', 'w').close() #reset size of dictionary
        self.training_hist=[]
        self.remaking=False
        self.count=0
        
        
        
        
        #create from a model file
        if len( name.split('.h5') ) >1:
            self.name = name.replace(".h5","").split("models")[1][1:]
            self.model = load_model(name)
            #rewrite dictionary
            file1=dir_path+'/dict.txt'
            file2=name.replace(".h5","")+'-dict.txt'
            with open(file2, 'rb') as f2, open(file1, 'wb') as f1:
                shutil.copyfileobj(f2, f1)
            
        #create from scratch
        else:
            if len( name.split('_v') )==1: self.name= name.split('.h5')[0] + '_v0.0'
            else: self.name=name
            self.model = Sequential()
            self.model.add(LSTM(128, input_shape=(num_words, 0), return_sequences=True))
            self.model.add(LSTM(128))
            self.model.add(Dense(0))
            self.model.add(Activation("softmax"))
    
    #return string of updated name past decimal
    #ex: 1.3 to 1.4
    def nameup_dot(self):
        self.name= f"{'_'.join(self.name.split('_')[:-1])}_{self.name.split('_')[-1].split('.')[0]}.{1+int(self.name.split('_')[-1].split('.')[1])}"
    #before dot
    def nameup_gen(self):
        self.name= f"{'_'.join(self.name.split('_')[:-1])}_{self.name.split('_')[-1].split('.')[0][0]}{1+int(self.name.split('_')[-1].split('.')[0][1:])}.{0}"
    def saveModel(self):
        #model
        print()
        try:
            os.remove(dir_path+'/models/'+self.name+'.h5')
            print("WARNING: saveModel: Model file already exists, overwriting")
        except OSError: pass
        self.model.save(dir_path+'/models/'+self.name+'.h5')
        #dict
        try:
            os.remove(dir_path+'/models/'+self.name+'-dict.txt')
            print("WARNING: saveModel: Dict file already exists, overwriting")
        except OSError: pass
        shutil.copyfile(dir_path+'/dict.txt', dir_path+'/models/'+self.name+'-dict.txt')
        
        

    #===========================--------------------------========================
    #TRAINING    
    def train(self,file,maxcount=None):
        #--------------
        #NOTE:get dictionary into an array
        dict=[]
        with open('dict.txt', 'r') as f:
            for line in f:
                #print(line.rstrip())
                dict.append(line.split("\n")[0])
        cnt= len(dict)
                
        #--------------
        #NOTE:get tokens of this dataset
        #text = red(file)
        text=[]
        with open(file, 'r') as f:
            for i in f.readline().split(' '):
                if not i or i.isspace() or "\n" in i: continue
                text.append( i.lower() )

        combined = list( np.concatenate((dict, text)) )
        #print(combined)

        # tokenizer = RegexpTokenizer(r"\w+")
        # tokens = tokenizer.tokenize(combined)
        unique_tokens = np.unique(combined)
        
        #--------------
        #NOTE:if theres more unique tokens, add to dict array, remake model
        if cnt != len(unique_tokens):
            self.count=0
            #------
            # remake dict
            #if not self.remaking:
            #print("hi1")
            with open('dict.txt', 'w') as f:
                for i in unique_tokens:
                    #f.write(f"{i}\t{unique_token_index[i]}\n")
                    f.write(f"{i}\n")
            self.training_hist.append(file)
                            
            #------
            #remake model
            self.model = Sequential()
            self.model.add(LSTM(128, input_shape=(self.num_words, len(unique_tokens)), return_sequences=True))
            self.model.add(LSTM(128))
            self.model.add(Dense(len(unique_tokens)))
            self.model.add(Activation("softmax"))
        
            #recursively retrain with entire history
            self.remaking=True
            self.nameup_gen()
            
            
            # print(f"DICT: {dict}")
            # print(f"{self.remaking},\t{cnt}\t{len(unique_tokens)}")
            # input("WAIT")
            
            
            for i in self.training_hist:
                self.train(i,maxcount)
            self.remaking=False
            self.saveModel()
            
                    
        #--------------
        #NOTE:else just train with data
        else:
            if maxcount: prPurple(f"<{self.count}/{maxcount}: {'{:.2f}'.format(float(100*self.count/maxcount))}%>\tTRAINING CURR:\t{file}...")
            else: prPurple(f"<{self.count}>\tTRAINING CURR:\t{file}...")
            #input("WAIT-new")
            
            unique_token_index = {token:idx for idx, token in enumerate(unique_tokens)}
            input_words = []
            next_words = []

            for i in range(len(combined) - self.num_words):
                input_words.append(combined[i:i +self.num_words])
                next_words.append(combined[i + self.num_words])
                
            X = np.zeros(  ( len(input_words), self.num_words, len(unique_tokens) ),  dtype=bool)
            Y = np.zeros(  ( len(next_words), len(unique_tokens)),  dtype=bool )
            
            for i, words in enumerate(input_words):
                for j, word in enumerate(words):
                    X[i, j, unique_token_index[word]] = 1
                Y[i, unique_token_index[next_words[i]]] = 1
            self.model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
            self.model.fit(X, Y, batch_size=128, epochs=30, shuffle=True, verbose=self.verbose)
            
            if not self.remaking:
                self.nameup_dot()
                self.saveModel()
            self.count+=1