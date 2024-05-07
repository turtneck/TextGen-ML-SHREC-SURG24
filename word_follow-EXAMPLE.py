#Tensor Flow, predict next word out of severl options, picking most likely at a time


import os
import random
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

print("imports pased...")




#-------------=============-------------=============-------------=============
#[!!!!!!!!!!!!]get datasets

global dir_path
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t{dir_path}>")

text_df = pd.read_csv(dir_path+"/data/fake_or_real_news.csv")
print(text_df)
text = list(text_df.text.values)
joined_text = " ".join(text)
print("AHHHHHHHHHHHHHH",len(joined_text))
partial_text = joined_text[:2269]
print("AHHHHHHHHHHHHHH",len(partial_text))

tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())
#print(tokens)

unique_tokens = np.unique(tokens)
unique_token_index = {token:idx for idx, token in enumerate(unique_tokens)}

num_words = 10
input_words = []
next_words = []

for i in range(len(tokens) - num_words):
    input_words.append(tokens[i:i +num_words])
    next_words.append(tokens[i + num_words])

#improvements:
'''
dont make all big list
keep a list of unique tokens that are added too
maybe start it with an import of a dictionary /\
save to a text file

have addition of data as a seperate file, keeping track of dictionary
-load last model
-create new dataset
-keep/add to dictionary
-save model_v<date>

function to export input_words,next_words given one string of text
with a giant collection of these arrays, train the ai
'''

#print(input_words)

X = np.zeros(  ( len(input_words), num_words, len(unique_tokens) ),  dtype=bool)
Y = np.zeros(  ( len(next_words), len(unique_tokens)),  dtype=bool )

#print(X)
#print(Y)

for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        X[i, j, unique_token_index[word]] = 1
    Y[i, unique_token_index[next_words[i]]] = 1



print("datasets passed...")

#-------------=============-------------=============-------------=============
#[!!!!!!!!!!!!]training

model = Sequential()
model.add(LSTM(128, input_shape=(num_words, len(unique_tokens)), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
model.fit(X, Y, batch_size=128, epochs=30, shuffle=True)

model.save('mymodel_WF_examplev2.h5')

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


print(  generate_text("He will have to look into this thing and he", 100, 5)  )

# print("\n========================")
# model.fit(X, Y, batch_size=128, epochs=30, shuffle=True)

# print(  generate_text("He will have to look into this thing and he", 100, 5)  )

print(unique_tokens)
print(unique_token_index)

with open('dict.txt', 'w') as f:
    for i in unique_tokens:
        #f.write(f"{i}\t{unique_token_index[i]}\n")
        f.write(f"{i}\n")
        
model.save(dir_path+'/models/'+'mymodel_WF_examplev2.h5')