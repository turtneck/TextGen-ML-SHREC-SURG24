import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
#------------------------
import os,sys,time,datetime,re,tiktoken
import numpy as np
from fun_colors import *
PTV2_HYPER_DEF=[24,128*2,0.7,10,1000,30000,100,1e-3,200,64,4,4,0.0]
print("import pass")
#------------------------------------------------

#MY BY-CHAR LIBRARY

#Final edition for pretraining
#Could add functionality of 'End of Sentence' after each '.!?' but not worth it right now (time)


#==========================================================
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, device, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.device = device
        self.block_size = block_size
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
print("general ML class pass")
#------------------------------------------------


#==========================================================
class PT_model_v3:
    def __init__(self, hyperparameters=PTV2_HYPER_DEF, model_path=None,name=None):
        # defaults ---------------------
        torch.manual_seed(1337)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        prPurple(self.device)
        if hyperparameters == None: raise SyntaxError("ERROR CREATING MODEL: MISSING INIT DATA")
                    
        # hyperparameters ---------------------
        self.batch_size =   hyperparameters[0] # how many independent sequences will we process in parallel?
        self.block_size =   hyperparameters[1] # max input/out len *2
        self.goal =         hyperparameters[2]
        self.goal_cnt =     hyperparameters[3]
        self.min_iters =    hyperparameters[4]
        self.max_iters =    hyperparameters[5]
        self.eval_interval= hyperparameters[6]
        self.learning_rate= hyperparameters[7]
        self.eval_iters =   hyperparameters[8]
        self.n_embd =       hyperparameters[9]
        self.n_head =       hyperparameters[10]
        self.n_layer =      hyperparameters[11]
        self.dropout =      hyperparameters[12]
        # self.hyperparameters = hyperparameters
            
            
        # meta data ---------------------
        self.vocab_size = tiktoken.get_encoding("gpt2").n_vocab+2 #gpt2 vocab size(+2 buffer,SOS) 'tiktoken.get_encoding("gpt2").n_vocab'
        # buffer ---------------------
        # the base dict is 0-50256: buffer 50257, SOS 50258
        self.buffer = self.vocab_size-2
        self.SOS = self.vocab_size-1
        # hypers ---------------------
        prGreen(hyperparameters)
        # name ---------------------
        if name is None: self.name = 'PTv3-tktk'
        else: self.name = 'PTv3-tktk_'+name
            
            
        # Model ---------------------
        if model_path == None:
            #make new model
            if hyperparameters == None: raise SyntaxError("ERROR CREATING MODEL: MISSING INIT DATA")
        
            self.model = BigramLanguageModel(device=self.device, vocab_size=self.vocab_size, block_size=self.block_size, n_embd=self.n_embd, n_head=self.n_head, n_layer=self.n_layer, dropout=self.dropout)
            self.m = self.model.to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            
            prGreen("SUCCESS: MODEL CREATED")
        elif model_path[-3:] !='.pt':
            #load latest model from a directory
            prGreen("Loading latest")
            prALERT("Please double check your   < hyperparameters >   are aligned with saved model")
            #!! get dir of models
            model_list = os.listdir(model_path)
            
            #!! load last model
            #   NOTE; each model name is saved in the format:      PTv(version)__(datetime)__(dataset index).pt
            #   this has it sorted by version, time created, dataset ran; each in accending order
            try:
                #find last model
                if len(model_list) == 0: raise IndexError("ERROR:  Model Directory is empty, no 'latest' models to choose from")
                if model_list[-1][-3:] != '.pt': raise IndexError(f"ERROR:  Latest 'Model' is invalid file type: < {model_list[-1][-3:]} >")
                
                prLightPurple(model_path+"/"+model_list[-1])
                self.model = BigramLanguageModel(device=self.device, vocab_size=self.vocab_size, block_size=self.block_size, n_embd=self.n_embd, n_head=self.n_head, n_layer=self.n_layer, dropout=self.dropout)
                self.model.load_state_dict(  torch.load(model_path+"/"+model_list[-1], map_location=self.device)  )
                self.model.eval()
                self.m = self.model.to(self.device)
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
                
                prGreen("SUCCESS: MODEL LOADED")
            except Exception as e:
                prALERT(str(e))
                print(Style.RESET_ALL)
                os._exit()         
        else:
            #load model from file
            # prALERT("Please double check your   < hyperparameters >   are aligned with saved model")
            # prLightPurple(model_path)
            # self.model = BigramLanguageModel(device=self.device, vocab_size=self.vocab_size, block_size=self.block_size, n_embd=self.n_embd, n_head=self.n_head, n_layer=self.n_layer, dropout=self.dropout)
            # self.model.load_state_dict(  torch.load(model_path, map_location=self.device)  )
            # self.model.eval()
            # self.m = self.model.to(self.device)
            # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            prALERT("Please double check your   < hyperparameters >   are aligned with saved model")
            prLightPurple(model_path)
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            self.m = self.model.to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            
            print("SUCCESS: MODEL LOADED")
            
    
    
    # ========================================
    def save_model(self,dir_path):
        torch.save(self.model, dir_path)
    
    
    # ========================================
    def load_model(self,dir_path):
        self.model = torch.load(dir_path)
        self.model.eval()
        self.m = self.model.to(self.device)
    
    
    # ========================================
    def run_model(self,data=None,length=256,verbose=False):
        if data is None:
            context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            target = self.m.generate(context, max_new_tokens=length)[0].tolist()
            target = self.PT_decode(target,verbose)
            return ('GEN:~'+target )
        else:
            data_yay = list( data_clean(data) )
            try:
                context = self.PT_encode2(data_yay)
                context.append(self.SOS)
            except KeyError as e:
                prRed(f"ERROR ENCODING INTO DICT:\t invalid key{e}")
                return
            
            context= self.PT_encode3([context])
            context = context.to(self.device)
            target = self.m.generate(context, max_new_tokens=length)[0].tolist()
            target = self.PT_decode(target,verbose)
            return (f'Q:~{data_clean(data)}\nA:~{target}' )
    
    def PT_encode(self,data):
        return torch.from_numpy( np.array(tiktoken.get_encoding("gpt2").encode(data), dtype=np.int64) ).type(torch.long)
    #split up versions of ^^^ for train_model_prompt2
    def PT_encode2(self,data):
        return tiktoken.get_encoding("gpt2").encode(data)
    def PT_encode3(self,data):
        return torch.from_numpy( np.array(data, dtype=np.int64) ).type(torch.long)
    
    def PT_decode(self,data,verbose=False):
        try:
            data=data[data.index(self.SOS)+1:]   #cut off
        except Exception as e:
            if verbose: prALERT(str(e))
            target_index = None #dont care about this error, if its not in it shouldn't be
        try:
            data=data[:data.index(self.buffer)+1]   #cut off
        except Exception as e:
            if verbose: prALERT(str(e))
            target_index = None #dont care about this error, if its not in it shouldn't be
        return tiktoken.get_encoding("gpt2").decode(data)
    
    # ==================================================================================================
    #NOTE: train model from the procedding character (finishing)
    #txt or int64 encoded bin array
    def train_model_basic(self,dir_path,savepath=None,logpath=None,start=0,end=None,add_message='',save_iter=1000,max_iters=None):
        if max_iters is None: max_iters = self.max_iters
        prGreen(f'train_dir: {dir_path}')
        prGreen(f'savepath: {savepath}')
        dirlist=os.listdir(dir_path)
        cnt=start
        
        if end:
            if end>len(dirlist)-1: raise ValueError("End past size of data")
            dirlist=dirlist[start:end]
            sze=end-1
        else:
            dirlist=dirlist[start:]
            sze=len(dirlist)-1
        
        if logpath==None: logpath = getDrive()+f'Model_Log/PyTorch/{self.name}-TRAIN__{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}.txt'
        prGreen(f'logpath: {logpath}')
        script_time=time.time()
        file_helper( logpath )#if log doesnt exist make it
        if self.hyperparameters: logger(logpath,   f"{self.hyperparameters}")
        else: logger(logpath,   f"No hyperparameters given during objects INIT")
        logger(logpath,   f"\n\n[!!!!!] START\t{str(datetime.datetime.now())}")
        
        for txtpath in dirlist:
            txt=dir_path+"/"+txtpath
            prCyan(add_message+f"PROG {cnt-start}/{sze+1}: <{gdFL( 100*(cnt-start)/(sze+1) )}%>\t{txt}...")
            logger(logpath,   add_message+f"PROG {cnt-start}/{sze+1}: <{gdFL( 100*(cnt-start)/(sze+1) )}%>\t{txt}...======================================")
            start_time=time.time()
            
            print( txtpath[-4:] )
            if txtpath[-4:] == '.txt':
                # print(".txt file")
                with open(txt, 'r', encoding="utf-8") as f: data = f.readlines()[1:-1]
                data = ''.join(data)
                #cleanup
                data = data_clean(data)
                train_data_torch = self.PT_encode(data)
            elif txtpath[-4:] == '.bin':
                # print(".bin file")
                train_data_torch = torch.from_numpy( np.fromfile(txt, dtype=np.int64) ).type(torch.long)
            else: raise TypeError(f"non 'txt' or 'bin' file for 'train_model_prompt' not supported")
            
            #actual training
            goal_cnt=0
            for iter in range(max_iters):
                # every once in a while evaluate the loss on train and val sets
                if iter % self.eval_interval == 0 or iter == max_iters - 1:
                    losses = self.estimate_loss(train_data_torch)
                    nowtime=time.time()
                    prYellow(add_message+f"PROG {cnt-start}/{sze+1}: <{gdFL( 100*(cnt-start)/(sze+1) )}%>\t<{gdFL( 100*iter/max_iters )}%>  step {iter}/{max_iters}:{' '*(2+len(str(max_iters))-len(str(iter)))}train loss {losses:.4f}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                    logger(logpath,   f"step {iter}:{' '*(2+len(str(max_iters))-len(str(iter)))}train loss {losses:.4f}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")

                # sample a batch of data
                xb, yb = self.get_batch(train_data_torch)

                # evaluate the loss
                logits, self.loss = self.model(xb, yb)
                self.optimizer.zero_grad(set_to_none=True)
                self.loss.backward()
                self.optimizer.step()
                if losses <= self.goal: goal_cnt+=1
                if goal_cnt>=self.goal_cnt: break
            #post
            nowtime=time.time()
            prPurple(add_message+f"end: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
            logger(logpath,   f"end: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
            
            #save
            if cnt%save_iter == 0:
                if savepath: self.save_model(savepath+f'{self.name}__{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}__{cnt}.pt')
                else: self.save_model(getDrive()+f'Models/PyTorch_v3/{self.name}__{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}__{cnt}.pt')
                nowtime=time.time()
                prLightPurple(add_message+f"save: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                logger(logpath,   f"save: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
            cnt+=1
    
    
    # ========================================
    #NOTE: train model from 'Q&A' Prompts
    #Trains with 'Q' as input and 'A' as target
    #csv (Q&A each <= 256 chars) - can be > but will skip over and waste time
    #DEFUCT!!!!!!!!!!!!!!!!
    '''
    def train_model_prompt(self,dir_path,savepath=None,logpath=None,start=0,end=None,add_message='',save_iter=1000,max_iters=None):
        if max_iters is None: max_iters = self.max_iters
        prGreen(f'train_dir: {dir_path}')
        prGreen(f'savepath: {savepath}')
        
        #NOTE: [!!!!] load csv, iterate through it for each training
        print( dir_path[-4:] )
        if dir_path[-4:] == '.csv':
            if end:
                if end>csv_size(dir_path): raise ValueError("End past size of data")
                sze=end-1
            else: sze = csv_size(dir_path)-start #get size of data (#rows)
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
        while True:
            try:
                if not end is None:
                    if cnt >= end: break
                prCyan(add_message+f"PROG {cnt-start}/{sze+1}: <{gdFL( 100*(cnt-start)/(sze+1) )}%>...")
                logger(logpath,   add_message+f"PROG {cnt-start}/{sze+1}: <{gdFL( 100*(cnt-start)/(sze+1) )}%>...======================================")
                start_time=time.time()                    
                
                df = next(df_iter)
                
                #annoying conversion from array of strings to an array of chars
                try:
                    question = list( list(df.question)[0] )
                    response = list( list(df.response)[0] )
                except Exception:
                    question = list( str(list(df.question)[0]) )
                    response = list( str(list(df.response)[0]) )
                question = list( data_clean(''.join(question)) )
                response = list( data_clean(''.join(response)) )
                
                if len(question) > self.block_size or len(response) > self.block_size:
                    prRed( f"Skipped {cnt}:\tq{len(question)}, r{len(response)}>{self.block_size}" )
                    logger(logpath, f"Skipped {cnt}:\tq{len(question)}, r{len(response)}>{self.block_size}")
                    cnt+=1
                    continue
                
                if len(question)<self.block_size:
                    for i in range( self.block_size-len(question) ): question.append('')
                if len(response)<self.block_size:
                    for i in range( self.block_size-len(response) ): response.append('')
                
                try:
                    train_torch_prompt = self.PT_encode(question)
                    train_torch_target = self.PT_encode(response)
                except KeyError as e:
                    prRed(f"Skipped {cnt}:\tERROR: invalid key{e}")
                    logger(logpath, f"Skipped {cnt}:\tERROR: invalid key{e}")
                    cnt+=1
                    continue
                del response,question
                
                
                #----------------------------------
                
                #actual training
                goal_cnt=0
                for iter in range(max_iters):
                    # every once in a while evaluate the loss on train and val sets
                    if iter % self.eval_interval == 0 or iter == max_iters - 1:
                        losses = self.estimate_loss(train_torch_prompt,train_torch_target)
                        nowtime=time.time()
                        prYellow(add_message+f"PROG {cnt-start}/{sze+1}: <{gdFL( 100*(cnt-start)/(sze+1) )}%>\t<{gdFL( 100*iter/max_iters )}%>  step {iter}/{max_iters}:{' '*(2+len(str(max_iters))-len(str(iter)))}train loss {losses:.4f}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                        logger(logpath,   f"step {iter}:{' '*(2+len(str(max_iters))-len(str(iter)))}train loss {losses:.4f}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")

                    # sample a batch of data
                    xb, yb = self.get_batch(train_torch_prompt,train_torch_target)

                    # evaluate the loss
                    logits, self.loss = self.model(xb, yb)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.loss.backward()
                    self.optimizer.step()
                    if losses <= self.goal: goal_cnt+=1
                    if goal_cnt>=self.goal_cnt: break
                #post
                nowtime=time.time()
                prPurple(add_message+f"end: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                logger(logpath,   f"end: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                
                #save
                if cnt%save_iter == 0:
                    if savepath: self.save_model(savepath+f'{self.name}__{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}__{cnt}.pt')
                    else: self.save_model(getDrive()+f'Models/PyTorch_v3/{self.name}__{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}__{cnt}.pt')
                    nowtime=time.time()
                    prLightPurple(add_message+f"save: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                    logger(logpath,   f"save: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                cnt+=1
            except StopIteration:
                break
    '''
    
    # ========================================
    #NOTE: train model from 'Q&A' Prompts
    #Trains based off a single string with a token specifying split between 'Q' and 'A'
    #csv
    def train_model_prompt2(self,dir_path,savepath=None,logpath=None,start=0,end=None,add_message='',save_iter=1000,max_iters=None):
        if max_iters is None: max_iters = self.max_iters
        prGreen(f'train_dir: {dir_path}')
        prGreen(f'savepath: {savepath}')
        
        #NOTE: [!!!!] load csv, iterate through it for each training
        print( dir_path[-4:] )
        if dir_path[-4:] == '.csv':
            if end:
                if end>csv_size(dir_path): raise ValueError("End past size of data")
                sze=end-1
            else: sze = csv_size(dir_path)-start #get size of data (#rows)
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
        while True:
            try:
                if not end is None:
                    if cnt >= end: break
                prCyan(add_message+f"PROG {cnt-start}/{sze+1}: <{gdFL( 100*(cnt-start)/(sze+1) )}%>...")
                logger(logpath,   add_message+f"PROG {cnt-start}/{sze+1}: <{gdFL( 100*(cnt-start)/(sze+1) )}%>...======================================")
                start_time=time.time()
                
                df = next(df_iter)
                
                #annoying conversion from array of strings to an array of chars
                try:
                    question = list( list(df.question)[0] )
                    response = list( list(df.response)[0] )
                except Exception:
                    question = list( str(list(df.question)[0]) )
                    response = list( str(list(df.response)[0]) )
                question = list( data_clean(''.join(question)) )
                response = list( data_clean(''.join(response)) )
                if len(question)>=1000 or len(response)>=1000:
                    prRed(f"Skipped {cnt}:\tERROR: too long {len(question)}, {len(response)}")
                    logger(logpath, f"Skipped {cnt}:\tERROR: too long {len(question)}, {len(response)}")
                    cnt+=1
                    continue
                
                
                try:
                    train_torch_input = self.PT_encode2(question)
                    train_torch_input.append(self.SOS)
                    train_torch_input.extend( self.PT_encode2(response) )
                    #if under 256, cant batch
                    if len(train_torch_input)<self.block_size:
                        for i in range( self.block_size-len(train_torch_input) ): train_torch_input.append(self.buffer)
                    train_torch_input = self.PT_encode3(train_torch_input)
                except KeyError as e:
                    prRed(f"Skipped {cnt}:\tERROR: invalid key{e}")
                    logger(logpath, f"Skipped {cnt}:\tERROR: invalid key{e}")
                    cnt+=1
                    continue
                del response,question
                
                #----------------------------------
                goal_cnt=0
                #actual training
                for iter in range(max_iters):
                    # every once in a while evaluate the loss on train and val sets
                    if iter % self.eval_interval == 0 or iter == max_iters - 1:
                        losses = self.estimate_loss(train_torch_input)
                        nowtime=time.time()
                        prYellow(add_message+f"PROG {cnt-start}/{sze+1}: <{gdFL( 100*(cnt-start)/(sze+1) )}%>\t<{gdFL( 100*iter/max_iters )}%>  step {iter}/{max_iters}:{' '*(2+len(str(max_iters))-len(str(iter)))}train loss {losses:.4f}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                        logger(logpath,   f"step {iter}:{' '*(2+len(str(max_iters))-len(str(iter)))}train loss {losses:.4f}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")

                    # sample a batch of data
                    xb, yb = self.get_batch(train_torch_input)

                    # evaluate the loss
                    logits, self.loss = self.model(xb, yb)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.loss.backward()
                    self.optimizer.step()
                    if losses <= self.goal: goal_cnt+=1
                    if goal_cnt>=self.goal_cnt: break
                #post
                nowtime=time.time()
                prPurple(add_message+f"end: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                logger(logpath,   f"end: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                
                #save
                if cnt%save_iter == 0:
                    if savepath: self.save_model(savepath+f'{self.name}__{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}__{cnt}.pt')
                    else: self.save_model(getDrive()+f'Models/PyTorch_3/{self.name}__{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}__{cnt}.pt')
                    nowtime=time.time()
                    prLightPurple(add_message+f"save: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                    logger(logpath,   f"save: {iter}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
                cnt+=1
            except StopIteration:
                break
    
    
    #====================================================================================================================
    #add target arg if training for prompts
    def get_batch(self,data, targets=None):
        # generate a small batch of data of inputs x and targets y
        if targets is None:
            if len(data) - self.block_size >0:ix = torch.randint( len(data) - self.block_size, (self.batch_size,))
            elif len(data) - self.block_size ==0:
                ix = [0]*self.batch_size
                data = torch.cat((data,torch.tensor([self.buffer])))
            else: raise ValueError("!!!![ERROR] get_batch(), target=None: len(data)<self.block_size")
            x = torch.stack([data[i:i+self.block_size] for i in ix])
            y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        else:
            x = torch.stack([data for i in range(self.batch_size)])
            y = torch.stack([targets for i in range(self.batch_size)])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    
    #add target arg if training for prompts
    @torch.no_grad()
    def estimate_loss(self,data, targets=None):
        self.model.eval()
        losses = torch.zeros(self.eval_iters)
        for k in range(self.eval_iters):
            X, Y = self.get_batch(data,targets)
            logits, self.loss = self.model(X, Y)
            losses[k] = self.loss.item()
        out = losses.mean()
        self.model.train()
        return out

print("tot ML class pass")
#------------------------------------------------


#==========================================================
VERSION = '3'
dir_path = os.path.abspath("")
modelname=''    #replace when ready

MODEL = PT_model_v3(
            model_path=dir_path+'/Models/'+modelname,
            name='_CRC-SFT2-type1')
print("Model create pass")
                    
#------------------------
#!! running
# prRed(f'TRAINING LEN: {len(os.listdir("book/gutenburg"))}')
# input("Ready to run training? <ENTER>")

#NOTE: TRAINING-------------------------
MODEL.train_model_prompt2(
    dir_path="prompt/1M-GPT4-Augmented_edit-256-1.csv",
    savepath=f"Models/PyTorch_v{VERSION}/SFT-type1/",
    logpath=f'Model_Log/PyTorch/Prompts/PTv{VERSION}_SFT-type1.txt',
    save_iter=100000,
    end=350000
    )
MODEL.save_model(f"Models/PyTorch_v{VERSION}/SFT-type1/PTv3-tktk__CRC-SFT2-type1__1M-GPT4.pt")

MODEL.train_model_prompt2(
    dir_path="prompt/3_5M-GPT3_5-Augmented_edit-256-1.csv",
    savepath=f"Models/PyTorch_v{VERSION}/SFT-type1/",
    logpath=f'Model_Log/PyTorch/Prompts/PTv{VERSION}_SFT-type1.txt',
    save_iter=100000,
    end=350000
    )
MODEL.save_model(f"Models/PyTorch_v{VERSION}/SFT-type1/PTv3-tktk__CRC-SFT2-type1__3_5M-GPT3_5.pt")

MODEL.train_model_prompt2(
    dir_path="prompt/MovieSorted-256-1.csv",
    savepath=f"Models/PyTorch_v{VERSION}/SFT-type1/",
    logpath=f'Model_Log/PyTorch/Prompts/PTv{VERSION}_SFT-type1.txt',
    save_iter=100000
    )
MODEL.save_model(f"Models/PyTorch_v{VERSION}/SFT-type1/PTv3-tktk__CRC-SFT2-type1__MovieSorted.pt")
