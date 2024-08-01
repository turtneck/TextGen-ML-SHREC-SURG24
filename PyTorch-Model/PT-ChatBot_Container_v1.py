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
PTChatBot_HYPER_DEF=[500,2,2,0.1,64,  50.0,1.0,0.0001,5.0,4000,1,500,  1000,0.7,10,30000]


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
        self.eval_interval  = hyperparameters[12]#from last containers
        self.goal           = hyperparameters[13]
        self.goal_cnt       = hyperparameters[14]
        self.max_iters      = hyperparameters[15]
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

        # If you have CUDA, configure CUDA to call
        if self.device == 'cuda':
            for state in self.encoder_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            for state in self.decoder_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        
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

        # If you have CUDA, configure CUDA to call
        if self.device == 'cuda':
            for state in self.encoder_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            for state in self.decoder_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        
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
        if max_iters is None: max_iters = self.max_iters
        prGreen(f'train_dir: {dir_path}')
        prGreen(f'savepath: {savepath}')
        
        #NOTE: [!!!!] load csv, iterate through it for each training
        print( dir_path[-4:] )
        if dir_path[-4:] == '.csv':
            sze = csv_size(dir_path)
            if sze < self.batch_size: raise ValueError(f"csv too small: {sze} < {self.batch_size}")
            if end:
                if end>sze: raise ValueError("End past size of data")
                sze=end-1
            else: sze = sze-start #get size of data (#rows)
            cnt=0
            df_iter = pd.read_csv(dir_path, iterator=True, chunksize=1)
            #iterate till start
            while cnt != start: df = next(df_iter); cnt+=1
        else: raise TypeError(f"nonCSV file for 'train_model_prompt' not supported")
        prGreen("CSV LOAD SUCCESS")
        
        #NOTE: [!!!!] setting uplog info
        if logpath==None: logpath = getDrive()+f'Model_Log/PyTorch/{self.name}-TRAIN__{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}.txt'
        prGreen(f'logpath: {logpath}')
        script_time=time.time()
        file_helper( logpath )#if log doesnt exist make it
        if self.hyperparameters: logger(logpath,   f"{self.hyperparameters}")
        else: logger(logpath,   f"No hyperparameters given during objects INIT")
        logger(logpath,   f"\n\n[!!!!!] START\t{str(datetime.datetime.now())}")
        
        
        #NOTE: [!!!!] iterate through dataset
        #start with collection of pairs
        input_batch = [None]#first val empty, load next 63 in; append and pop every cycle
        output_batch = [None]
        try:
            while len(input_batch)<64:
                df = next(df_iter)

                question = data_clean(''.join(str(list(df.question)[0])))
                response = data_clean(''.join(str(list(df.response)[0])))
                if len(question)>=1000 or len(response)>=1000:
                    prRed(f"Skipped {cnt}:\tERROR: too long {len(question)}, {len(response)}")
                    logger(logpath, f"Skipped {cnt}:\tERROR: too long {len(question)}, {len(response)}")
                    cnt+=1
                    continue
                #encode
                try:
                    question = self.PT_encode2(question)
                    question.append(self.EOS_token)
                    response = self.PT_encode2(response)
                    response.append(self.EOS_token)
                except KeyError as e:
                    prRed(f"Skipped {cnt}:\tERROR: invalid key{e}")
                    logger(logpath, f"Skipped {cnt}:\tERROR: invalid key{e}")
                    cnt+=1
                    continue
                #ready
                input_batch.append(question)
                output_batch.append(response)
        except StopIteration:
            raise ValueError(f"PRELOAD BREAK; csv too small: {sze} < {self.batch_size}")
        
        
        
        #------------
        #True Loop
        while True:
            try:
                if not end is None:
                    if cnt >= end: break
                prCyan(add_message+f"PROG {cnt-start}/{sze+1}: <{gdFL( 100*(cnt-start)/(sze+1) )}%>...")
                logger(logpath,   add_message+f"PROG {cnt-start}/{sze+1}: <{gdFL( 100*(cnt-start)/(sze+1) )}%>...======================================")
                start_time=time.time()
                
                #prep
                df = next(df_iter)

                question = data_clean(''.join(str(list(df.question)[0])))
                response = data_clean(''.join(str(list(df.response)[0])))
                if len(question)>=1000 or len(response)>=1000:
                    prRed(f"Skipped {cnt}:\tERROR: too long {len(question)}, {len(response)}")
                    logger(logpath, f"Skipped {cnt}:\tERROR: too long {len(question)}, {len(response)}")
                    cnt+=1
                    continue                
                #encode
                try:
                    question = self.PT_encode2(question)
                    question.append(self.EOS_token)
                    response = self.PT_encode2(response)
                    response.append(self.EOS_token)
                except KeyError as e:
                    prRed(f"Skipped {cnt}:\tERROR: invalid key{e}")
                    logger(logpath, f"Skipped {cnt}:\tERROR: invalid key{e}")
                    cnt+=1
                    continue
                #QUEUE Q&A
                input_batch.pop(0);input_batch.append(question)
                output_batch.pop(0);output_batch.append(response)
                
                #batch2TrainData
                #inputVar
                # prCyan( f'q: <<<{question}>>>, {type(question)}' )
                lengths = torch.tensor([len(indexes) for indexes in input_batch])
                input_variable = torch.LongTensor( list(itertools.zip_longest(*input_batch, fillvalue=self.PAD_token)) )
                # prPurple(f'\ninputVar_lengths: {lengths[0]}\n{type(lengths[0])}, {type(lengths)}, {lengths.shape}')
                # prPurple(f'\ninputVar_padVar: {input_variable[0]}\n{type(input_variable[0])}, {type(input_variable)}, {input_variable.shape}')
                
                #outputVar
                # prALERT('---------------')
                # prCyan( f'r: <<<{response}>>>, {type(response)}' )
                max_target_len = max([len(indexes) for indexes in output_batch])
                mask = torch.BoolTensor( self.binaryMatrix(list(itertools.zip_longest(*output_batch, fillvalue=self.PAD_token))) )
                target_variable = torch.LongTensor( list(itertools.zip_longest(*output_batch, fillvalue=self.PAD_token)) )
                # prPurple(f'\noutputVar_max_target_len: {max_target_len}\n{type(max_target_len)}')
                # prPurple(f'\noutputVar_mask: {mask[0]}\n{type(mask[0])}, {type(mask)}, {mask.shape}')
                # prPurple(f'\noutputVar_padVar: {target_variable[0]}\n{type(target_variable[0])}, {type(target_variable)}, {target_variable.shape}')
                
                del response,question
                                
                #----------------------------------
                self.encoder.train()
                self.decoder.train()
                goal_cnt=0
                
                #actual training
                for iter in range(max_iters):
                    #TODO: TRAINING
                    #===================================================
                    # Zero gradients
                    self.encoder_optimizer.zero_grad()
                    self.decoder_optimizer.zero_grad()

                    # Set device options
                    # prCyan(f'InVar: [{input_variable.shape}],\n{input_variable}')
                    input_variable = input_variable.to(self.device)
                    target_variable = target_variable.to(self.device)
                    mask = mask.to(self.device)
                    # Lengths for RNN packing should always be on the CPU
                    lengths = lengths.to("cpu")

                    # Initialize variables
                    loss = 0
                    print_losses = []
                    n_totals = 0

                    # Forward pass through encoder
                    encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

                    # Create initial decoder input (start with SOS tokens for each sentence)
                    decoder_input = torch.LongTensor([[self.SOS_token for _ in range(self.batch_size)]])
                    decoder_input = decoder_input.to(self.device)

                    # Set initial decoder hidden state to the encoder's final hidden state
                    decoder_hidden = encoder_hidden[:self.decoder.n_layers]

                    # Determine if we are using teacher forcing this iteration
                    use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

                    # Forward batch of sequences through decoder one time step at a time
                    # print(f'E_in:~{input_variable}')
                    # print(f'D_in:~{decoder_input}')
                    if use_teacher_forcing:
                        decoder_output, decoder_hidden = self.decoder( decoder_input, decoder_hidden, encoder_outputs )
                        # Teacher forcing: next input is current target
                        decoder_input = target_variable[0].view(1, -1)
                        # Calculate and accumulate loss
                        mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_variable[0], mask[0])
                        loss += mask_loss
                        print_losses.append(mask_loss.item() * nTotal)
                        n_totals += nTotal
                    else:
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                        # No teacher forcing: next input is decoder's own current output
                        _, topi = decoder_output.topk(1)
                        decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                        decoder_input = decoder_input.to(self.device)
                        # Calculate and accumulate loss
                        mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_variable[0], mask[0])
                        loss += mask_loss
                        print_losses.append(mask_loss.item() * nTotal)
                        n_totals += nTotal

                    # Perform backpropagation
                    loss.backward()

                    # Clip gradients: gradients are modified in place
                    _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
                    _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

                    # Adjust model weights
                    self.encoder_optimizer.step()
                    self.decoder_optimizer.step()

                    losses = sum(print_losses) / n_totals   #return: average loss
                    
                    
                    
                    
                    #===================================================
                    #return of training
                    #----
                    if iter % self.eval_interval == 0 or iter == max_iters - 1:
                        nowtime=time.time()
                        prYellow(add_message+f"PROG {cnt-start}/{sze+1}: <{gdFL( 100*(cnt-start)/(sze+1) )}%>\t<{gdFL( 100*iter/max_iters )}%>  step {iter}/{max_iters}:{' '*(2+len(str(max_iters))-len(str(iter)))}train loss {losses:.4f}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                        logger(logpath,   f"step {iter}:{' '*(2+len(str(max_iters))-len(str(iter)))}train loss {losses:.4f}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                    if losses <= self.goal: goal_cnt+=1
                    if goal_cnt>=self.goal_cnt: break
                #post
                nowtime=time.time()
                prPurple(add_message+f"end: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                logger(logpath,   f"end: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                
                #save
                # if cnt%save_iter == 0:
                #     if savepath: self.save_model(savepath+f'{self.name}__{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}__{cnt}.tar')
                #     else: self.save_model(getDrive()+f'Models/PT-ChatBot/{self.name}__{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}__{cnt}.tar')
                #     nowtime=time.time()
                #     prLightPurple(add_message+f"save: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                #     logger(logpath,   f"save: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                cnt+=1
            except StopIteration:
                break
        
    # ========================================
    def binaryMatrix(self, l):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == self.PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m
    def maskNLLLoss(self,inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(self.device)
        return loss, nTotal.item()
    def batch2TrainData(self,question,response):
        #inputVar
        # prCyan( f'q: <<<{question}>>>, {type(question)}' )
        lengths = torch.tensor([len(question)])
        input_variable = torch.LongTensor( list(itertools.zip_longest(*([question]*self.batch_size), fillvalue=self.PAD_token)) )
        # prPurple(f'\ninputVar_lengths: {lengths[0]}\n{type(lengths[0])}, {type(lengths)}, {lengths.shape}')
        # prPurple(f'\ninputVar_padVar: {input_variable[0]}\n{type(input_variable[0])}, {type(input_variable)}, {input_variable.shape}')
        
        #outputVar
        # prALERT('---------------')
        # prCyan( f'r: <<<{response}>>>, {type(response)}' )
        max_target_len = len(response)
        mask = torch.BoolTensor( self.binaryMatrix(list(itertools.zip_longest(*([response]*self.batch_size), fillvalue=self.PAD_token))) )
        target_variable = torch.LongTensor( list(itertools.zip_longest(*[response], fillvalue=self.PAD_token)) )
        # prPurple(f'\noutputVar_max_target_len: {max_target_len}\n{type(max_target_len)}')
        # prPurple(f'\noutputVar_mask: {mask[0]}\n{type(mask[0])}, {type(mask)}, {mask.shape}')
        # prPurple(f'\noutputVar_padVar: {target_variable[0]}\n{type(target_variable[0])}, {type(target_variable)}, {target_variable.shape}')
        return input_variable, lengths, target_variable, mask, max_target_len
    

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
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
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
    #general test----------
    # mod= PT_Chatbot(name='Test')
    # # # mod.run_model()
    
    # #general training test----------
    # mod.train_model(
    #     dir_path=getDrive()+"prompt/1M-GPT4-Augmented_edit-256-2.csv",
    #     logpath=getDrive()+f'CHATBOT-testing.txt',
    #     max_iters=1,
    #     end=1
    #     )
    
    #save/load test
    # mod.save_model(getDrive()+'Models/PT-ChatBot/PT-CB_SL-TEST.tar')
    # mod2 = PT_Chatbot(
    #     name='Test2',
    #     model_path=getDrive()+'Models/PT-ChatBot/PT-Chatbot-v1_2024-07-28_4_29__2024-07-28_4_32__2000.tar'
    #     )
    # mod2.train_model(
    #     dir_path=getDrive()+"prompt/1M-GPT4-Augmented_edit-256-2.csv",
    #     logpath=getDrive()+f'CHATBOT-testing.txt',
    #     max_iters=1,
    #     end=1
    #     )
    # mod2.run_model()
    
    
    
    
    
    #NOTE: CRC-------------------------
    # mod= PT_Chatbot()
    # print("Model create pass")
    
    # mod.train_model(
    #     dir_path="prompt/1M-GPT4-Augmented_edit-full-1.csv",
    #     savepath=f"Models/PT-ChatBot/",
    #     logpath=f'Model_Log/PT-ChatBot/PT-ChatBot_1M-GPT4.txt',
    #     save_iter=100000,
    #     end=350000
    #     )
    # mod.save_model(f"Models/PT-ChatBot/PT-ChatBot_1M-GPT4.tar")
    
    
    #----
    mod= PT_Chatbot(model_path="Models/PT-ChatBot/PT-ChatBot_1M-GPT4.tar")
    print("Model create pass")

    mod.train_model(
        dir_path="prompt/3_5M-GPT3_5-Augmented_edit-full-1.csv",
        savepath=f"Models/PT-ChatBot/",
        logpath=f'Model_Log/PT-ChatBot/PT-ChatBot_ChatBot_3_5M-GPT3_5.txt',
        save_iter=100000,
        end=350000
        )
    mod.save_model(f"Models/PT-ChatBot/PT-ChatBot_3_5M-GPT3_5.tar")

    mod.train_model(
        dir_path="prompt/MovieSorted-full-1.csv",
        savepath=f"Models/PT-ChatBot/",
        logpath=f'Model_Log/PT-ChatBot/PT-ChatBot_MovieSorted.txt',
        save_iter=100000
        )
    mod.save_model(f"Models/PT-ChatBot/PT-ChatBot_MovieSorted.tar")
