

import csv,os,sys,time,numpy,datetime,multiprocessing,re
import numpy as np
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path[:-5])
from fun_colors import *

# printpath=(getDrive()+"book/")
# dstr=f"{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}"
# script_time=time.time()
# file_helper( printpath+f'gutenburg_log-RBT-word__BIN.txt' )#if log doesnt exist make it
# logger(printpath+f'gutenburg_log-RBT-word__BIN.txt',   f"\n\n[!!!!!] START\t{str(datetime.datetime.now())}")

#------------------------
#load dict as sorted list
printpath=(getDrive()+"book/")
arr= sorted_byVAL(printpath+f'gutenDICT-RBT/char/gutenburg_dict-RBT-char.bin')
# print(arr)


#------------------------
#load meta
#meta = { 'vocab_size': len(arr), 'itos': itos, 'stoi': stoi, 'uint': 32 }
with open(printpath+ 'gutenburg_BIN\metas\gutenburg_bin-RBT-char_meta_int64.pkl', 'rb') as f: meta = pickle.load(f)
print(meta['vocab_size'])
print(meta['int'])
# print(np.array(meta['itos']))
# print(np.array(meta['stoi']))

#///////////////////////////////////////////////////////////////
#NOTE: manuals

start=0

#------------------------
BINdir=getDrive()+"book\\gutenburg_BIN\\char_64"
dirlist=os.listdir(BINdir)
sze=len(dirlist)-1
cnt=start
# print(dirlist[start:])
#open up all files
# input('Ready for spam???')
try:
    for txtpath in dirlist[start:1]:
        dat = np.fromfile(BINdir+'/'+txtpath, dtype=np.int64)
        print(txtpath, dat, len(dat))
except Exception as e:
    prALERT(f"wrd_e:\t\t<{txtpath}>")
    prALERT(e)