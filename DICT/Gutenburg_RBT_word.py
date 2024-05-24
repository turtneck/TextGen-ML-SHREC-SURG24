'''
BOILERPLATE:

Reads from a bunch of .txt files in a directory and adds words to a Red-Black Tree, adding the frequency each word appears.
Made specifically for the Gutenburg dataset.

This method is RAM heavy, and 100000x faster then the 'minRAM' approach that tries to make RAM use a little as possible
'''
#///////////////////////////////////////////////////////////////
#imports
import csv,os,sys,time,numpy,datetime,multiprocessing
import numpy as np
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path[:-5])
from fun_colors import *

#file path and global variables
script_time=time.time()
print(f"DRIVE_DIR:\t\t<{getDrive()+'book/gutenburg'}>")
printpath=(getDrive()+"book/")
file_helper( printpath+f'gutenburg_log-RBT-word.txt' )#if log doesnt exist make it
logger(printpath+f'gutenburg_log-RBT-word.txt',   f"\n\n[!!!!!] START\t{str(datetime.datetime.now())}")


#loading past dict---------------------
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/Datatypes')
from RBTree import RBT
print(f"DICT_FILE:\t\t<{ printpath+'gutenDICT-RBT/word/gutenburg_dict-RBT-word.bin' }>")
file_helper( printpath+'gutenDICT-RBT/word/gutenburg_dict-RBT-word.bin' )#if dict doesnt exist make it
RBTree = RBT( printpath+'gutenDICT-RBT/word/gutenburg_dict-RBT-word.bin' )

dstr=f"{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}"
fail=False
#///////////////////////////////////////////////////////////////
#NOTE: manuals

start=0

#------------------------
dirlist=os.listdir(getDrive()+"book\\gutenburg")
sze=len(dirlist)-1
cnt=start
#open up all files
try:
    for txtpath in dirlist[start:]:
        word_cnt=0;word_tot=0;last_word="";nospace=False
        txt=getDrive()+"book\\gutenburg"+"\\"+txtpath
        prCyan(f"PROG {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}...")
        
        
        #load whole data set into RAM (one big string) and format it down to words
        start_time=time.time()
        with open(getDrive()+"book\\gutenburg"+"\\"+txtpath, 'r', encoding="utf-8") as f: data = f.read()
        for i in ['™']: data=data.replace(i,"")
        for i in [',','--','---','[',']',';','*','•',':','"','“','”','(',')','&','=','�','—','\t','/','\\','_','|','<','>','\n','~']: data=data.replace(i," ")
        for i in ['***','?','!']: data=data.replace(i,".")
        for i in ['.\n', '. ', '..',".'"]: data=data.replace(i," ")
        data= data.split(" ")
        word_tot = len(data)
        word_cnt = RBTree.size
        
        # res,ind = np.unique(data, return_index=True)
        # data = res[np.argsort(ind)]
        # del res;del ind
        for wrd in data:
            if wrd=='' or len(wrd)<1: continue
            if wrd[0] == '-': wrd=wrd[1:]
            if wrd=='' or len(wrd)<1: continue
            if wrd[0] == "'" and wrd[-1] == "'": wrd=wrd[1:-1]
            elif wrd[0] == "'": wrd=wrd[1:]
            elif wrd[0] == "‘" and wrd[-1] == "’": wrd=wrd[1:-1]
            elif wrd[0] == "‘": wrd=wrd[1:]
            if wrd=='' or len(wrd)<1: continue
            RBTree.insert(wrd.lower())
        
        word_cnt=RBTree.size-word_cnt
        # if cnt%100 ==0: RBTree.save_tree(printpath+'gutenDICT-RBT/word/gutenburg_dict-RBT-word_t.bin')
        nowtime=time.time()
        prYellow( f"{  goodtime(nowtime-start_time)  }\t+{word_cnt}/{word_tot} <{gdFL(100*word_cnt/word_tot)}%> words\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
        t_str=f"PROG {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>  {txt}..."
        logger(printpath+f'gutenburg_log-RBT-word.txt',   f"{t_str}{'.'*(55-len(t_str))}\t{  goodtime(nowtime-start_time)  }\t+{word_cnt}/{word_tot} <{gdFL(100*word_cnt/word_tot)}%> words\t<{   goodtime(nowtime-script_time)   }> RUNTIME\t{last_word}")
        cnt+=1
        #------------------
except Exception as e:
    fail=True
    nowtime=time.time()
    logger(printpath+f'gutenburg_log-RBT-word.txt',   f"FAILLLLLLL PROG<> {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}...\t{  goodtime(nowtime-start_time)  }\t+{word_cnt}/{word_tot} <{gdFL(100*word_cnt/word_tot)}%> words\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
    prALERT(f"data:\t\t{data}")
    prALERT(f"RBTree.size:\t\t{RBTree.size}")
    prALERT(f"wrd:\t\t<{wrd}> {len(wrd)} {len(wrd)<1}")
    RBTree.save_tree(printpath+f'gutenDICT-RBT/word/gutenburg_dict-RBT-word_FAIL__{dstr}.bin')
    prALERT(e)
    
    

#///////////////////////////////////////////////////////////////
if not fail:
    RBTree.save_tree(printpath+f'gutenDICT-RBT/word/gutenburg_dict-RBT-word__{dstr}.bin')
    RBTree.save_tree(printpath+'gutenDICT-RBT/word/gutenburg_dict-RBT-word.bin')
    print( RBTree.inorder_arr_VAL() )
