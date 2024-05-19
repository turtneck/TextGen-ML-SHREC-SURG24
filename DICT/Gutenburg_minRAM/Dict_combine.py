import csv,os,sys,time,numpy,datetime,multiprocessing
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))[:-5]
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path)
from fun_colors import *


printpath=(getDrive()+"book\\")
print(printpath)

for i in range(10):
    print(printpath+f"gutenburg_dict{i}.txt")
    
    #go through each num dict, combine into tot dict
    dict_txt = open(printpath+f'gutenburg_dict{i}.txt','r', encoding="utf-8")
    while True:
        wrd= dict_txt.readline()
        if wrd == "": break
        if "\n" in wrd:wrd=wrd[:-1]
        else: wrd=wrd
        
        
        inside=False    
        #with open('temp_dict.txt','r') as read_queue:
        read_queue = open(printpath+'gutenburg_dict.txt','r', encoding="utf-8")
        #reads each line at a time instead of usual array iteration method, less memory required
        while True:
            t_read= read_queue.readline()
            if t_read == "": break
            if "\n" in t_read:line=t_read[:-1]
            else: line=t_read
            if wrd.lower() == line: inside=True; break
        read_queue.close()
        #NOTE: if not inside, add to file
        if not inside:
            read_queue= open(printpath+'gutenburg_dict.txt','a', encoding="utf-8")
            read_queue.write(wrd.lower()+"\n")
            read_queue.close()