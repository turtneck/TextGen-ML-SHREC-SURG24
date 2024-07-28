#https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
#Encoder to Decoder(w/ Attn Layer) Transformer

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import unicodedata
import codecs
from io import open
import itertools
import math
import json
#------------------------
import os,sys,time,datetime,re,tiktoken
import numpy as np
import pandas as pd
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.abspath("")
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path)
from fun_colors import *
#------------------------
PTChatBot_HYPER_DEF=[500,2,2,0.1,64,  50.0,1.0,0.0001,5.0,4000,1,500]


#===============================================================================
class PT_Chatbot:
    def __init__(self, meta_data=None, hyperparameters=PTChatBot_HYPER_DEF, model_path=None,name=None,buffer=None,attn_model='dot'):
        torch.manual_seed(1337)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        prPurple(self.device)
        
        # hyperparameters ---------------------
        #config
        self.hidden_size        = hyperparameters[0]
        self.encoder_n_layers   = hyperparameters[1]
        self.decoder_n_layers   = hyperparameters[2]
        self.dropout            = hyperparameters[3]
        self.batch_size         = hyperparameters[4]
        self.attn_model         = attn_model    #dot, general, concat
        #training
        self.clip                   = hyperparameters[5]
        self.teacher_forcing_ratio  = hyperparameters[6]
        self.learning_rate          = hyperparameters[7]
        self.decoder_learning_ratio = hyperparameters[8]
        self.n_iteration            = hyperparameters[9]
        self.print_every            = hyperparameters[10]
        self.save_every             = hyperparameters[11]
        self.hyperparameters = hyperparameters
            
            
        # meta data ---------------------
        if meta_data:
            self.dict_type=False #custom
            with open(meta_data, 'rb') as f: meta = pickle.load(f)
            self.stoi = meta['stoi']
            self.itos = meta['itos']
            self.vocab_size = meta['vocab_size']+2
            if meta['int'] == 8: self.mdtype=np.int8
            elif meta['int'] == 16: self.mdtype=np.int16
            elif meta['int'] == 32: self.mdtype=np.int32
            elif meta['int'] == 64: self.mdtype=np.int64
            elif meta['int'] == 128: self.mdtype=np.int128
            elif meta['int'] == 256: self.mdtype=np.int256
            else: raise TypeError(f"unknown meta data type signed: {meta['int']}")
            # buffer ---------------------
            # in my dict 7925 or '' is buffer, 7926 is SOS
            if buffer is None: self.PAD_token = [self.stoi[c] for c in ['']][0]
            else: self.PAD_token = buffer
            self.SOS_token = self.vocab_size-2
            self.EOS_token = self.vocab_size-1
        else:
            self.dict_type=True #tiktoken
            #tiktoken
            self.vocab_size = tiktoken.get_encoding("gpt2").n_vocab+3 #gpt2 vocab size(+2 buffer,SOS) 'tiktoken.get_encoding("gpt2").n_vocab'
            # buffer ---------------------
            # the base dict is 0-50256: buffer 50257, SOS 50258
            self.PAD_token = self.vocab_size-3
            self.SOS_token = self.vocab_size-2
            self.EOS_token = self.vocab_size-1
            
        
        # hypers ---------------------
        prGreen(hyperparameters)
        
        # name ---------------------
        if name is None: self.name = 'PT-Chatbot-v1'
        else: self.name = 'PT-Chatbot-v1_'+name
            
            
        # Model ---------------------
        if model_path:
            prYellow('Loading modelpath info')
            # If loading on same machine the model was trained on
            checkpoint = torch.load(model_path)
            # If loading a model trained on GPU to CPU
            #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            encoder_sd = checkpoint['en']
            decoder_sd = checkpoint['de']
            encoder_optimizer_sd = checkpoint['en_opt']
            decoder_optimizer_sd = checkpoint['de_opt']
            embedding_sd = checkpoint['embedding']
            
        prYellow('Building encoder and decoder ...')
        #Init emmbeddings
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        if model_path: self.embedding.load_state_dict(embedding_sd)
        
        #Init encoder/decoder
        self.encoder = EncoderRNN(self.hidden_size, self.embedding, self.encoder_n_layers, self.dropout)
        self.decoder = LuongAttnDecoderRNN(self.attn_model, self.embedding, self.hidden_size, self.vocab_size, self.decoder_n_layers, self.dropout)
        if model_path:
            self.encoder.load_state_dict(encoder_sd)
            self.decoder.load_state_dict(decoder_sd)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        # Initialize optimizers
        prYellow('Building optimizers ...')
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate * self.decoder_learning_ratio)
        if model_path:
            self.encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            self.decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        
        self.encoder.eval()
        self.decoder.eval()
        
        self.Model = GreedySearchDecoder(self.encoder, self.decoder,self.device,self.SOS_token)
        #------
        prGreen("SUCCESS: MODEL LOADED")
            
    
    
    # ========================================
    def save_model(self,dir_path):
        torch.save({
                    'en': self.encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': self.encoder_optimizer.state_dict(),
                    'de_opt': self.decoder_optimizer.state_dict(),
                    'embedding': self.embedding.state_dict()
                    }, dir_path)
    
    
    # ========================================
    def load_model(self,dir_path):
        prYellow('Loading modelpath info')
        # If loading on same machine the model was trained on
        checkpoint = torch.load(dir_path)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
            
        prYellow('Building encoder and decoder ...')
        #Init emmbeddings
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embedding.load_state_dict(embedding_sd)
        
        #Init encoder/decoder
        self.encoder = EncoderRNN(self.hidden_size, self.embedding, self.encoder_n_layers, self.dropout)
        self.decoder = LuongAttnDecoderRNN(self.attn_model, self.embedding, self.hidden_size, self.vocab_size, self.decoder_n_layers, self.dropout)
        self.encoder.load_state_dict(encoder_sd)
        self.decoder.load_state_dict(decoder_sd)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        # Initialize optimizers
        prYellow('Building optimizers ...')
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate * self.decoder_learning_ratio)
        self.encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        self.decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        
        self.encoder.eval()
        self.decoder.eval()
        
        self.Model = GreedySearchDecoder(self.encoder, self.decoder,self.device,self.SOS_token)
        #------
        prGreen("SUCCESS: MODEL LOADED")
    
    
    # ========================================
    def run_model(self,length=None,verbose=False):
        if length is None:
            if self.dict_type:length=10
            else:length=256
        
        self.encoder.eval()
        self.decoder.eval()

        while(True):
            try:
                # Get input sentence
                input_sentence = input('> ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit': break
                
                #!encode input + EOS char
                context = data_clean(input_sentence) # Normalize sentence
                try:
                    context = self.PT_encode2(context)
                    context.append(self.EOS_token)
                except KeyError as e:
                    raise ValueError(f"ERROR ENCODING INTO DICT:\t invalid key{e}")
                context= torch.LongTensor([context]).transpose(0, 1)
                context = context.to(self.device)
                
                #create 1D tensor of len(input)
                lengths = torch.tensor([len(indexes) for indexes in [context]]) #create 1D tensor of len(input)
                
                
                #!generate
                tokens, scores = self.Model(context, lengths, length)
                #!decode
                decoded_words = self.PT_decode([token.item() for token in tokens],True)
                               
                
                # Format and print response sentence
                print('Bot:', decoded_words)
            except KeyError:
                print("Error: Encountered unknown word.")
    
    '''def evaluate(self,input_data,max_length=256):
        #encode input + EOS char
        try:
            input_batch = self.PT_encode2(input_data)
            input_batch.append(self.EOS_token)
        except KeyError as e:
            raise ValueError(f"ERROR ENCODING INTO DICT:\t invalid key{e}")
        input_batch= self.PT_encode3([input_batch])
        input_batch = input_batch.to(self.device)
        #create 1D tensor of len(input)
        lengths = torch.tensor([len(indexes) for indexes in input_batch])
    
    
        #generate
        tokens, scores = self.Model(input_batch, lengths, max_length)
        #decode
        decoded_words = self.PT_decode([token.item() for token in tokens])
        return decoded_words'''
    
    
    # ========================================
    def PT_encode(self,data):
        if self.dict_type: return torch.from_numpy( np.array(tiktoken.get_encoding("gpt2").encode(data), dtype=np.int64) ).type(torch.long)
        else: return torch.from_numpy( np.array(fun_encode(data, self.stoi), dtype=np.int64) ).type(torch.long)
    #split up versions of ^^^ for train_model_prompt2
    def PT_encode2(self,data):
        if self.dict_type: return tiktoken.get_encoding("gpt2").encode(data)
        else: return fun_encode(data, self.stoi)
    def PT_encode3(self,data):
        if self.dict_type: return torch.from_numpy( np.array(data, dtype=np.int64) ).type(torch.long)
        else: return torch.from_numpy( np.array(data, dtype=np.int64) ).type(torch.long)

    def PT_decode(self,data,verbose=False):
        # try:
        #     data=data[data.index(self.SOS_token)+1:]   #cut off
        # except Exception as e:
        #     if verbose: prALERT(str(e))
        #     target_index = None #dont care about this error, if its not in it shouldn't be
        # try:
        #     data=data[:data.index(self.PAD_token)+1]   #cut off
        # except Exception as e:
        #     if verbose: prALERT(str(e))
        #     target_index = None #dont care about this error, if its not in it shouldn't be
        if self.dict_type: return tiktoken.get_encoding("gpt2").decode(data)
        else: return fun_decode(data,self.itos)
    
    # ==================================================================================================
    def train_model(self,dir_path,savepath=None,logpath=None,start=0,end=None,add_message='',save_iter=1000,max_iters=None):
        return
    

#==========================================================
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size parameters are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
#------
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
#--------------
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
#--------------
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, device, SOS_token):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device=device
        self.SOS_token=SOS_token

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * self.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
    
    
#==========================================================
if __name__ == "__main__":
    mod= PT_Chatbot(name='Test')
    mod.run_model()