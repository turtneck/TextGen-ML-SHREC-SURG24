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
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path[:-5])
from fun_colors import *

#file path and global variables
script_time=time.time()
print(f"DRIVE_DIR:\t\t<{getDrive()+'book/gutenburg'}>")
printpath=(getDrive()+"book/dict/")
log_name=printpath+'gutenburg_log-RBT-chr-TEST.txt'
file_helper( log_name )#if log doesnt exist make it
logger(log_name,   f"\n\n[!!!!!] START\t{str(datetime.datetime.now())}")


#loading past dict---------------------
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/Datatypes')
dict_name=printpath+'gutenburg_dict-RBT-chr-TEST.bin'
from RBTree import RBT
print(f"DICT_FILE:\t\t<{ dict_name }>")
file_helper( dict_name )#if dict doesnt exist make it
dict = RBT( dict_name )

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
        last_word="";nospace=False
        txt=getDrive()+"book\\gutenburg"+"\\"+txtpath
        prCyan(f"PROG {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}...")
        
        
        #load whole data set into RAM (one big string) and format it down to words
        start_time=time.time()
        with open(getDrive()+"book\\gutenburg"+"\\"+txtpath, 'r', encoding="utf-8") as f: data = f.readlines()[1:-1]
        
        #tokenize
        res,ind = np.unique(data, return_index=True)
        data = res[np.argsort(ind)]
        del res;del ind
        
        word_cnt = dict.size
        #cleanup
        for wrd in data:
            for i in ['™']: wrd=wrd.replace(i,"")
            for i in ['“','”']: wrd=wrd.replace(i,'"')
            for i in ['‘','’']: wrd=wrd.replace(i,"'")
            for i in ['--','---','***','�','—','\t','_','|']: wrd=wrd.replace(i," ")
            wrd= re.sub(' {2,}',' ',wrd)
            #if wrd in ['',' ',' \n','\n']: continue
            if wrd=='' or len(wrd)<1: continue
            for chr in wrd: dict.insert(chr)
        
        word_cnt=dict.size-word_cnt
        # if cnt%100 ==0: RBTree.save_tree(printpath+'gutenDICT-RBT/char/gutenburg_dict-RBT-chr_t.bin')
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
    prALERT(f"data:\t\t{data}")
    prALERT(f"dict size:\t\t{dict.size}")
    dict.save_tree(printpath+f'gutenDICT-RBT/char/gutenburg_dict-RBT-chr_FAIL__{dstr}.bin')
    prALERT(e)
    
    

#///////////////////////////////////////////////////////////////
if not fail:
    dict.save_tree(printpath+f'gutenDICT-RBT/char/gutenburg_dict-RBT-chr__{dstr}.bin')
    dict.save_tree(printpath+'gutenDICT-RBT/char/gutenburg_dict-RBT-chr.bin')
    print( dict.inorder_arr_VAL() )