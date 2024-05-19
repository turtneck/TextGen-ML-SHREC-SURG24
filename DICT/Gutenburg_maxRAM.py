'''
BOILERPLATE:

Reads from a bunch of .txt files in a directory and place the unique words in a csv file,
this file is the "DICT_Queue" that needs to be *manually* picked apart if a word is "proper" for the dict model to train on.

This does NOT check if a word is proper or has invalid characters inside the word.
That is the work of the DICT model trained on this data.
It will however filter out specific characters in the 'clean' list object that interupt the flow of characters.

Some of this is overdone, this was done on purpose to be reused later in other models
'''
#///////////////////////////////////////////////////////////////
#imports
import csv,os,sys,time,numpy,datetime,multiprocessing
import numpy as np
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))[:-5]
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path)
from fun_colors import *

#file path and global variables
script_time=time.time()
print(f"DRIVE_DIR:\t\t<{getDrive()+'book/gutenburg'}>")
printpath=(getDrive()+"book/")


#loading past dict---------------------
file_helper(printpath+f'gutenburg_dict_RAM+.bin')#if dict doesnt exist make it


#///////////////////////////////////////////////////////////////
#NOTE: manuals

start=0

#------------------------
dirlist=os.listdir(getDrive()+"book\\gutenburg")
sze=len(dirlist)
cnt=start
#open up all files
try:
    for txtpath in dirlist:
        word_cnt=0;word_tot=0;last_word="";nospace=False
        txt=getDrive()+"book\\gutenburg"+"\\"+txtpath
        prCyan(f"PROG<> {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}...")
        
        
        #load whole data set into RAM (one big string) and format it down to words
        start_time=time.time()
        with open(getDrive()+"book\\gutenburg"+"\\"+txtpath, 'r', encoding="utf-8") as f: data = f.read()
        for i in ['™']: data=data.replace(i,"")
        for i in [',','--','---','[',']',';','*','•',':','"','“','”','(',')','&','=','�','—','\t','/','\\','_','|','<','>','\n']: data=data.replace(i," ")
        for i in ['***','?','!']: data=data.replace(i,".")
        for i in ['.\n', '. ']: data=data.replace(i," ")
        data= list(np.unique( data.split(" ") ))
        data11=[]
        for wrd in data:
            if wrd=='':continue
            if wrd[0] == "'" and wrd[-1] == "'": wrd=wrd[1:-1]
            elif wrd[0] == "'": wrd=wrd[1:]
            elif wrd[0] == "‘" and wrd[-1] == "’": wrd=wrd[1:-1]
            elif wrd[0] == "‘": wrd=wrd[1:]
            data11.append(wrd)
        data = data11.copy(); del data11
        
        
        
        
                
        nowtime=time.time()
        prYellow( f"{  goodtime(nowtime-start_time)  }\t+{word_cnt}/{word_tot} <{gdFL(100*word_cnt/word_tot)}%> words\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
        t_str=f"PROG<> {start+cnt}/{end}: <{gdFL( 100*cnt/end )}%>\t{txt}..."
        logger(printpath+f'gutenburg_log.txt',   f"{t_str}{'.'*(55-len(t_str))}\t{  goodtime(nowtime-start_time)  }\t+{word_cnt}/{word_tot} <{gdFL(100*word_cnt/word_tot)}%> words\t<{   goodtime(nowtime-script_time)   }> RUNTIME\t{last_word}")
        cnt+=1
        #input("B")
        #break
except Exception as e:
    nowtime=time.time()
    logger(printpath+f'gutenburg_log.txt',   f"FAILLLLLLL PROG<> {start+cnt}/{end}: <{gdFL( 100*cnt/end )}%>\t{txt}...\t{  goodtime(nowtime-start_time)  }\t+{word_cnt}/{word_tot} <{gdFL(100*word_cnt/word_tot)}%> words\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
    prALERT(f"te:\t\t{te}")
    prALERT(f"words:\t\t{words}")
    prALERT(f"wrd:\t\t{wrd}")
    prALERT(e)
#///////////////////////////////////////////////////////////////

