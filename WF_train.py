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