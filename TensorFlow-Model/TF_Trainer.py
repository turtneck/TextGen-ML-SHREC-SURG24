from fun_colors import *
import os,time,sys
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

#me :D
from WF_create import WF_model

global dir_path
dir_path = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__": print(f"DIRECTORY:\t\t{dir_path}>")



'''
NOTE NOTE NOTE NOTE NOTE NOTE NOTE
This is ripped, needs to be made actually functional
TODO
- needs adjustment to look at all .txt files in '\DATA' directory
- put these directories into an array
    - have directories only have text files
- make 'latest' go through this like 'all known'
- make 'all known's array be these directories
    - start with this version and copy over to 'latest'
'''

#=======================================================
if __name__ == "__main__":
    print(Back.CYAN+"1:Latest  "+Style.RESET_ALL)
    print(Back.CYAN+"2:All Known Data "+Style.RESET_ALL)
    inp0= int(input(Fore.CYAN+"SELECT #: "));print(Style.RESET_ALL)
    
    if inp0 == 1:
        dir_list = os.listdir(dir_path+'/models')
        arr=[]
        for i in dir_list:
            if (".h5" in i) and (i.replace('.h5','')+'-dict.txt'in dir_list): arr.append(i)
        arr = sorted(arr, key=len)
        model = WF_model(dir_path+'/models/'+arr[-1])
        
        limit=1000
        cur=0;start_t=0;pat=dir_path+'\data\init_news/'

        #----------------------------------------------
        #actually running
        limit=1000
        cur=0;start_t=0
        
        start_tot=time.time()
        dir_list = os.listdir(pat)
        
        model.training_hist=[pat+s for s in dir_list[:len(arr)]]
        print(model.training_hist)
        
        prRed(Back.RED+f"CURR LIST: {dir_list[len(arr)]}...{dir_list[limit-1]}"+Style.RESET_ALL)
        prRed(Back.RED+f"LEN: {len(dir_list[len(arr):limit])}"+Style.RESET_ALL)
        print(Back.RED+f"ROUGH EST TIME: {limit*28}sec"+Style.RESET_ALL)
        input(Fore.CYAN+"Ready to run training? <ENTER>")
        
        for data in dir_list[len(arr):limit]:
            prGreen("\n----------------------------------------------")
            prGreen(f"<{cur}/{limit-len(arr)}: {'{:.2f}'.format(float(100*cur/(limit-len(arr))))}%>\tSTARTING: {dir_path+i+data}")
            
            start_t= time.time()
            model.train(pat+data,cur+1+len(arr))
            print(Back.GREEN+f"[Training total return] runtime: {time.time()-start_t}"+Style.RESET_ALL)
            cur+=1
        
        print("\n\n")
        print(Back.CYAN+"=============-------------=============-------------=============")
        print(f"<{cur}/{limit}> DONE!!!!")
        print(f"Total Runtime:\t{time.time()-start_tot}");sys.stdout.write(Style.RESET_ALL)
        
    else:
        model = WF_model("WF_jeb")
        limit=3
        cur=0;start_t=0

        #list of dataset folders
        dataset_paths=['\data\init_news/']

        #----------------------------------------------
        word_lis=[]
        for i in dataset_paths:
            dir_list = os.listdir(dir_path+i)
            # print(dir_list)
            #print(dir_list[:limit])
            for data in dir_list[:limit]:
                #print(dir_path+i+data)
                with open(dir_path+i+data, 'r') as f:
                    for j in f.readline().split(' '):
                        word_lis.append( j )

        prRed(f"LENGTH:\t{len(word_lis)}")



        #----------------------------------------------
        #actually running
        start_tot=time.time()
        for i in dataset_paths:
            dir_list = os.listdir(dir_path+i)
            
            #prRed(f"CURR LIST:\t{dir_list[:limit]}")
            prRed(Back.RED+f"CURR LIST: {dir_list[0]}...{dir_list[limit-1]}"+Style.RESET_ALL)
            prRed(Back.RED+f"LEN: {len(dir_list[:limit])}"+Style.RESET_ALL)
            print(Back.RED+f"ROUGH EST TIME: {limit*28}sec"+Style.RESET_ALL)
            input(Fore.CYAN+"Ready to run training? <ENTER>")
            
            for data in dir_list[:limit]:
                prGreen("\n----------------------------------------------")
                prGreen(f"<{cur}/{limit}: {'{:.2f}'.format(float(100*cur/limit))}%>\tSTARTING: {dir_path+i+data}")
                
                start_t= time.time()
                model.train(dir_path+i+data,cur+1)
                print(Back.GREEN+f"[Training total return] runtime: {'{:.5f}'.format(time.time()-start_t)}"+Style.RESET_ALL)
                cur+=1
            
            print("\n\n")
            print(Back.CYAN+"=============-------------=============-------------=============")
            print(f"<{cur}/{limit}> DONE!!!!")
            print(f"Total Runtime:\t{time.time()-start_tot}");sys.stdout.write(Style.RESET_ALL)