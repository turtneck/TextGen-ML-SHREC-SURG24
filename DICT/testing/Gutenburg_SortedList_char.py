'''
BOILERPLATE:

Reads from a bunch of .txt files in a directory and adds characters to a Red-Black Tree, adding the frequency each character appears.
Made specifically for the Gutenburg dataset.

This method is RAM heavy, and 100000x faster then the 'minRAM' approach that tries to make RAM use a little as possible
'''
#///////////////////////////////////////////////////////////////
#imports
import csv,os,sys,time,numpy,datetime,multiprocessing,re
import numpy as np
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.abspath("")
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path)
from fun_colors import *

#file path and global variables
script_time=time.time()
print(f"DRIVE_DIR:\t\t<{getDrive()+'book/gutenburg'}>")
printpath=(getDrive()+"book/dict/")
log_name=printpath+'gutenburg_log-SortList-chr-TEST.txt'
print(f"LOG_DIR:\t\t<{log_name}>")
file_helper( log_name )#if log doesnt exist make it
logger(log_name,   f"\n\n[!!!!!] START\t{str(datetime.datetime.now())}")


#loading past dict---------------------
#///////////////////////////////////////////////////////////////



class SortList():
    def __init__(self):
        self.arr = []
        self.size=0

    def insert(self, key):
        if not key in self.arr:
            self.size+=1
            index=0 #least-max list
            for i in self.arr:
                if key<i:break
                index+=1
            self.arr.insert(index,key)
        return



#///////////////////////////////////////////////////////////////
dict = SortList()

dstr=f"{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}"
fail=False
#///////////////////////////////////////////////////////////////
#NOTE: manuals

start=0

#------------------------
dirlist=os.listdir(getDrive()+"book/gutenburg")
sze=len(dirlist)-1
cnt=start
#open up all files
try:
    for txtpath in dirlist[start:]:
        last_word="";nospace=False
        txt=getDrive()+"book/gutenburg/"+txtpath
        prCyan(f"PROG {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}...")
        
        
        #load whole data set into RAM (one big string) and format it down to words
        start_time=time.time()
        with open(getDrive()+"book/gutenburg/"+txtpath, 'r', encoding="utf-8") as f: data = f.readlines()[1:-1]
        data = ''.join(data)
        data=data_clean(data)
        word_cnt = dict.size
        for chr in data: dict.insert(chr)
        
        word_cnt=dict.size-word_cnt
        nowtime=time.time()
        prYellow( f"{  goodtime(nowtime-start_time)  }\t+<{word_cnt}> chars\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
        t_str=f"PROG {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>  {txt}..."
        logger(log_name,   f"{t_str}{'.'*(56-len(t_str))}\t+<{word_cnt}> chars\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME\t{last_word}")
        cnt+=1
        #------------------
except Exception as e:
    fail=True
    nowtime=time.time()
    logger(log_name,   f"FAILLLLLLL PROG<> {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}...\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
    prALERT(f"dict size:\t\t{dict.size}")
    prALERT(e)
    
    

#///////////////////////////////////////////////////////////////
if not fail: print("bro passed")
else: print(f"bro failed:\t{cnt}\t{txt}")