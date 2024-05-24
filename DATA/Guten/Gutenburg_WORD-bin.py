# Training prep, requires dictionary

import csv,os,sys,time,numpy,datetime,multiprocessing,re
import numpy as np
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path[:-11])
from fun_colors import *

printpath=(getDrive()+"book/")
dstr=f"{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}"
script_time=time.time()
file_helper( printpath+f'gutenburg_log-RBT-word__BIN.txt' )#if log doesnt exist make it
logger(printpath+f'gutenburg_log-RBT-word__BIN.txt',   f"\n\n[!!!!!] START\t{str(datetime.datetime.now())}")

#------------------------
#load dict as sorted list
printpath=(getDrive()+"book/")
arr= sorted_byVAL(printpath+f'gutenDICT-RBT/word/gutenburg_dict-RBT-word.bin')
print(arr[:25])


#------------------------
#enumerate list
stoi = { ch:i for i,ch in enumerate(arr) }
itos = { i:ch for i,ch in enumerate(arr) }

meta = { 'vocab_size': len(arr), 'itos': itos, 'stoi': stoi, 'uint': 32 }
with open(printpath+ 'gutenburg_BIN\metas\gutenburg_bin-RBT-word_meta.pkl', 'wb') as f: pickle.dump(meta, f)
with open(printpath+ f'gutenburg_BIN\metas\gutenburg_bin-RBT-word_meta__{dstr}.pkl', 'wb') as f: pickle.dump(meta, f)
del arr

prYellow(f"ENCODE/DECODE OVERHEAD:\t{ goodtime(time.time()-script_time) }")
logger(printpath+f'gutenburg_log-RBT-word__BIN.txt',   f"ENCODE/DECODE OVERHEAD:\t{ goodtime(time.time()-script_time) }")
# print("\n\n",stoi);print("\n\n",itos)
#------------------------
#go through data set, converting each

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
        txt=getDrive()+"book\\gutenburg"+"\\"+txtpath
        prCyan(f"PROG {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}...")
        
        
        #load whole data set into RAM (one big string) and format it down to chrs
        start_time=time.time()
        with open(getDrive()+"book\\gutenburg"+"\\"+txtpath, 'r', encoding="utf-8") as f: data = f.readlines()[1:-1]
        data = ' '.join(data)
        for i in ['™']: data=data.replace(i,"")
        for i in [',','--','---','[',']',';','*','•',':','"','“','”','(',')','&','=','�','—','\t','/','\\','_','|','<','>','\n','~']: data=data.replace(i," ")
        for i in ['***','?','!']: data=data.replace(i,".")
        for i in ['.\n', '. ', '..',".'"]: data=data.replace(i," ")
        data= data.split(" ")
        
        data1=[]
        for wrd in data:
            if wrd=='' or len(wrd)<1: continue
            if wrd[0] == '-': wrd=wrd[1:]
            if wrd=='' or len(wrd)<1: continue
            if wrd[0] == "'" and wrd[-1] == "'": wrd=wrd[1:-1]
            elif wrd[0] == "'": wrd=wrd[1:]
            elif wrd[0] == "‘" and wrd[-1] == "’": wrd=wrd[1:-1]
            elif wrd[0] == "‘": wrd=wrd[1:]
            if wrd=='' or len(wrd)<1: continue
            # if wrd.lower() not in arr: prALERT(f"<{wrd.lower()}>")
            data1.append(wrd.lower())
        data=data1[:]; del data1
        # data = ' '.join(data)
        # print(data)
        train_ids = encode(data, stoi)
        # print(train_ids[:25])
        train_ids = np.array(train_ids, dtype=np.uint32)
        # print(train_ids[:25])
        train_ids.tofile(getDrive()+f"book\\gutenburg_BIN\word\GB_pg{int(txt[20:-4])}.bin")
        
        
        # if cnt%100 ==0: RBTree.save_tree(printpath+'gutenDICT-RBT/word/gutenburg_dict-RBT-word_t.bin')
        nowtime=time.time()
        prYellow( f"{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME\t{getDrive()+f'book/gutenburg_BIN/word/GB_pg{int(txt[20:-4])}.bin'}")
        t_str=f"PROG {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>  {txt}..."
        logger(printpath+f'gutenburg_log-RBT-word__BIN.txt',   f"{t_str}{'.'*(56-len(t_str))}\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME\t{getDrive()+f'book/gutenburg_BIN/word/GB_pg{int(txt[20:-4])}.bin'}")
        cnt+=1
        #------------------
except Exception as e:
    prALERT(f"wrd_e:\t\t<{wrd}>")
    nowtime=time.time()
    logger(printpath+f'gutenburg_log-RBT-word__BIN.txt',   f"FAILLLLLLL PROG<> {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}...\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME\t{getDrive()+f'book/gutenburg_BIN/word/GB_pg{int(txt[20:-4])}.bin'}")
    prALERT(f"data:\n{data}")
    prALERT(e)