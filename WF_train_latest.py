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

#=======================================================
if __name__ == "__main__":
    
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