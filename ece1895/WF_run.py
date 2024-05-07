from fun_colors import *
import os,time,random,sys
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

global model,unique_tokens,unique_token_index,dir_path
dir_path = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__": print(f"DIRECTORY:\t\t{dir_path}>")

num_words=10
tokenizer = RegexpTokenizer(r"\w+")

def predict_nextword(input_text, n_best):
    input_text = input_text.lower()
    X = np.zeros(  (1, num_words, len(unique_tokens))  )
    for i, word in enumerate(input_text.split()):
        X[0, i, unique_token_index[word]] = 1
    
    predictions = model.predict(X)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]

# possible = predict_nextword("He will have to look into this thing and he", 5)
# print(unique_tokens[idx] for idx in possible)

def generate_text(input_text, text_length, creativity=3):
    word_sequence = input_text.split()
    current = 0
    for _ in range(text_length):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+num_words])
        try:
            choice = unique_tokens[random.choice(predict_nextword(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)


def load_model_nice(name):
    global unique_tokens,unique_token_index,model
    
    model = load_model(dir_path+'/models/'+name)
    unique_tokens=[]
    with open(dir_path+'/models/'+name.replace('.h5','')+'-dict.txt', 'r') as f:
        for line in f:
            unique_tokens.append(line.split("\n")[0])
    unique_token_index = {token:idx for idx, token in enumerate(unique_tokens)}


#================================================================================
#================================================================================
if __name__ == "__main__":
    
    print(Back.CYAN+"1:Choice  "+Style.RESET_ALL)
    print(Back.CYAN+"2:All Run "+Style.RESET_ALL)
    print(Back.CYAN+"3:Latest  "+Style.RESET_ALL)
    inp0= int(input(Fore.CYAN+"SELECT #: "));print(Style.RESET_ALL)
    
    
    #list all models
    dir_list = os.listdir(dir_path+'/models')
    
    if inp0 == 1:
        prCyan(Fore.CYAN+"Avaliable Models: ")
        arr=[]
        for i in dir_list:
            if (".h5" in i) and (i.replace('.h5','')+'-dict.txt'in dir_list): arr.append(i)
        arr = sorted(arr, key=len)
        for i in range(len(arr)): prLightGray(f"<{i+1}>: {arr[i]}")
        
        #-------------------
        inp= input(Back.GREEN+"SELECT #: ");print(Style.RESET_ALL)
        load_model_nice(arr[int(inp)-1])
        
        #-------------------
        inp1=input(Back.GREEN+"#words created??: ");print(Style.RESET_ALL)
        if inp1: inp1 = int(inp1); inp2 = int(input(Back.GREEN+"creativity??: "));print(Style.RESET_ALL)
        else: inp1=100; inp2=1
            
        #-------------------
        sys.stdout.write(Fore.RED+"Enter text:");sys.stdout.write(Style.RESET_ALL)
        inp3=input(" ")
        if not inp3: inp3="He will have to look into this thing and he"
        print(  generate_text( inp3, inp1, inp2)  )
    
    elif inp0 == 2:
        #AUTOMATED. NOT USER
        arr=[]
        for i in dir_list:
            if (".h5" in i) and (i.replace('.h5','')+'-dict.txt'in dir_list): arr.append(i)
        arr = sorted(arr, key=len)
        for i in arr:
            load_model_nice(i)
            print("\n-------------------")
            print( f'<{i}>\n{generate_text( "He will have to look into this thing and he" , 100, 1)}' )
            input("WAIT")
            
    else:
        arr=[]
        for i in dir_list:
            if (".h5" in i) and (i.replace('.h5','')+'-dict.txt'in dir_list): arr.append(i)
        arr = sorted(arr, key=len)
        load_model_nice(arr[-1])
        print(  generate_text( "He will have to look into this thing and he", 100, 1)  )