import torch
import torch.nn as nn
from torch.nn import functional as F
#------------------------
import os,sys,time,datetime
import numpy as np
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.abspath("")
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path)
from fun_colors import *
from PT_Container_v1 import PT_model_v1, PTV1_HYPER_DEF
#------------------------

for i in range(1,33):
    mod = PT_model_v1(getDrive()+"book/gutenburg_BIN\metas\gutenburg_bin-RBT-char_meta_int64.pkl",[i,32,0.7,1000,30000,100,1e-3,200,64,4,4,0.0])
    mod.train_model(getDrive()+"book\\gutenburg_BIN\\char_64",logpath=getDrive()+f'Model_Log\PyTorch\PTv1_Threads\\PTv1_paraTrain_{i}.txt',end=1)