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
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))[:-5]
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path)
from fun_colors import *

#file path and global variables
#///////////////////////////////////////////////////////////////
def partition(parts):
    part=57588/parts
    arr=[]
    last=0
    for i in range(parts):
        arr.append([last,  int(part*(i+1)-1)  ])
        last=int(part*i)
    return arr
#------------------------
def Gutenburg_dict(num=0,start=0,end=57587):
    prALERT(f"THREAD<<{num}<< STARTING")
    print(f"DRIVE_DIR:\t\t<{getDrive()+'book/gutenburg'}>")
    printpath=(getDrive()+"book\\gutenburg").split("\\")[0]+"\\"+(getDrive()+"book\\gutenburg").split("\\")[1]+"\\"

    logger(printpath+f'gutenburg_log{num}.txt',   f"\n\n[!!!!!] START\t{str(datetime.datetime.now())}")
    script_time=time.time()
    
    #if dict doesnt exist make it
    t_file= open(printpath+f'gutenburg_dict{num}.txt','a', encoding="utf-8")
    t_file.close()
    del t_file
    
    dirlist=os.listdir(getDrive()+"book\\gutenburg")[start:end]
    cnt=0
    #open up all files
    try:
        for txtpath in dirlist:
            word_cnt=0;word_tot=0;last_word="";nospace=False
            txt=getDrive()+"book\\gutenburg"+"\\"+txtpath
            #sys.stdout.write(f'{Fore.CYAN}{txt}...'+Style.RESET_ALL)
            #prCyan(txt+"...")
            prCyan(f"PROG<{num}> {start+cnt}/{end}: <{gdFL( 100*cnt/end )}%>\t{txt}...")
            #iterate over each word
            start_time=time.time()
            with open(getDrive()+"book\\gutenburg"+"\\"+txtpath, 'r', encoding="utf-8") as f:
                #NOTE: find start of book
                te=""
                while "*** START OF THE PROJECT GUTENBERG" not in te: te=f.readline()
                #NOTE:add to temp string until has a ".", try to add before words to "DICT_Queue"
                te=""
                #for line in f:
                while True:
                    line= f.readline()
                    if line == "</pre></body></html>":
                        # words = [x for x in i.split(" ") if x]#remove empty spaces
                        # for wrd in words:
                        #     addword(wrd)
                        break
                    #if "*** END OF THE PROJECT GUTENBERG" in line: break
                    #NOTE:cleanup
                    if not line or line.isspace(): continue
                    #remove starting spaces from line
                    nonspc=0
                    for i in range(len(line)):
                        if line[i] != " ": nonspc=i;break
                    line=line[i:]               
                    if "\n" in line:
                        line=line[:-1]
                        if nospace:
                            te+=line
                            nospace=False
                        elif line[-1] == '-':
                            nospace=True
                            te+=" "+line
                        else: te+=" "+line
                    else:
                        if nospace:
                            te+=line
                            nospace=False
                        elif line[-1] == '-':
                            nospace=True
                            te+=line
                        else: te+=line
                    for i in [',','--','---','[',']',';','*','™','•',':','"','“','”','(',')','&','=','�','—','\t','/','\\','_','|']: te=te.replace(i," ")
                    for i in ['***','?','!']: te=te.replace(i,".")
                    
                    #NOTE:check for sentences
                    #print(f"<{te}>")
                    if "." in te:
                        #print(te.split("."))
                        for i in te.split(".")[:-1]:
                            words = [x for x in i.split(" ") if x]#remove empty spaces
                            #print(words)
                            for wrd in words:
                                #NOTE: check if word in file
                                inside=False
                                word_tot+=1
                                
                                #single
                                if wrd[0] == "'" and wrd[-1] == "'": wrd=wrd[1:-1]
                                elif wrd[0] == "'": wrd=wrd[1:]
                                elif wrd[0] == "‘" and wrd[-1] == "’": wrd=wrd[1:-1]
                                elif wrd[0] == "‘": wrd=wrd[1:]
                                #double
                                # elif wrd[0] == '"' and wrd[-1] == '"': wrd=wrd[1:-1]
                                # elif wrd[0] == '"': wrd=wrd[1:]
                                # elif wrd[0] == '“' and wrd[-1] == '”': wrd=wrd[1:-1]
                                # elif wrd[0] == '“': wrd=wrd[1:]
                                
                                #with open('temp_dict.txt','r') as read_queue:
                                read_queue = open(printpath+f'gutenburg_dict{num}.txt','r', encoding="utf-8")
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
                                    word_cnt+=1
                                    last_word=wrd.lower()
                                    read_queue= open(printpath+f'gutenburg_dict{num}.txt','a', encoding="utf-8")
                                    read_queue.write(wrd.lower()+"\n")
                                    read_queue.close()
                                #======================
                        te=te.split(".")[-1]
                    
                    #input("A")
            nowtime=time.time()
            prYellow( f"{  goodtime(nowtime-start_time)  }\t+{word_cnt}/{word_tot} <{gdFL(100*word_cnt/word_tot)}%> words\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
            t_str=f"PROG<{num}> {start+cnt}/{end}: <{gdFL( 100*cnt/end )}%>\t{txt}..."
            logger(printpath+f'gutenburg_log{num}.txt',   f"{t_str}{'.'*(55-len(t_str))}\t{  goodtime(nowtime-start_time)  }\t+{word_cnt}/{word_tot} <{gdFL(100*word_cnt/word_tot)}%> words\t<{   goodtime(nowtime-script_time)   }> RUNTIME\t{last_word}")
            cnt+=1
            #input("B")
            #break
    except Exception as e:
        nowtime=time.time()
        logger(printpath+f'gutenburg_log{num}.txt',   f"FAILLLLLLL PROG<{num}> {start+cnt}/{end}: <{gdFL( 100*cnt/end )}%>\t{txt}...\t{  goodtime(nowtime-start_time)  }\t+{word_cnt}/{word_tot} <{gdFL(100*word_cnt/word_tot)}%> words\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
        prALERT(f"te:\t\t{te}")
        prALERT(f"words:\t\t{words}")
        prALERT(f"wrd:\t\t{wrd}")
        prALERT(e)
    prALERT(f"=====================THREAD{num} END=====================")
#///////////////////////////////////////////////////////////////

class guten_dict_thr:
    def __init__(self,num,start,end):
        self.thread = multiprocessing.Process(target=Gutenburg_dict, args=(num, start, end))
        self.thread.start()

if __name__ == "__main__":
    # t1 = multiprocessing.Process(target=Gutenburg_dict, args=(0,    0,19195))
    # t2 = multiprocessing.Process(target=Gutenburg_dict, args=(1,19196,38391))
    # t3 = multiprocessing.Process(target=Gutenburg_dict, args=(2,38392,57587))

    # prALERT("THREAD STARTING")
    # t1.start()
    # t2.start()
    # t3.start()

    # t1.join()
    # t2.join()
    # t3.join()
    
    num_threads=10
    
    threads=[]
    list_arr=partition(num_threads)
    for thr in range(num_threads):
        threads.append(   guten_dict_thr(thr,list_arr[thr][0],list_arr[thr][1])   )
        
    for thr in threads:
        thr.thread.join()