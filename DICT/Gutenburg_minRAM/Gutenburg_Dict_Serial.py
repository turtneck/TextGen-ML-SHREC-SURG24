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
import csv,os,sys,time,numpy,datetime
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))[:-5]
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path)
from fun_colors import *

#file path and global variables
filepath = getDrive()+"book\\gutenburg"
print(f"DRIVE_DIR:\t\t<{filepath}>")
printpath=filepath.split("\\")[0]+"\\"+filepath.split("\\")[1]+"\\"
nospace=False #word-word between a new space consideration
global word_cnt,word_tot,last_word
logger(printpath+'gutenburg_log.txt',   f"\n\n[!!!!!] START\t{str(datetime.datetime.now())}")
script_time=time.time()

#///////////////////////////////////////////////////////////////
#NOTE: manuals
clean=[',','--','---','[',']',';','*','™','•',':','"','“','”','(',')','&','=','�','—','\t','/','\\']#spaces
clean2=['***','?','!']#replace with '.'s
clean3=['_','|']#replace space

start=0

#///////////////////////////////////////////////////////////////
def addword(str):
    #NOTE: check if word in file
    inside=False
    global word_cnt, word_tot, last_word
    word_tot+=1
    
    #single
    if str[0] == "'" and str[-1] == "'": str=str[1:-1]
    elif str[0] == "'": str=str[1:]
    elif str[0] == "‘" and str[-1] == "’": str=str[1:-1]
    elif str[0] == "‘": str=str[1:]
    #double
    # elif str[0] == '"' and str[-1] == '"': str=str[1:-1]
    # elif str[0] == '"': str=str[1:]
    # elif str[0] == '“' and str[-1] == '”': str=str[1:-1]
    # elif str[0] == '“': str=str[1:]
    
    #with open('temp_dict.txt','r') as read_queue:
    read_queue = open(printpath+'gutenburg.txt','r', encoding="utf-8")
    #reads each line at a time instead of usual array iteration method, less memory required
    while True:
        t_read= read_queue.readline()
        if t_read == "": break
        if "\n" in t_read:line=t_read[:-1]
        else: line=t_read
        if str.lower() == line: inside=True; break
    read_queue.close()
    #NOTE: if not inside, add to file
    if not inside:
        word_cnt+=1
        last_word=str.lower()
        read_queue= open(printpath+'gutenburg.txt','a', encoding="utf-8")
        read_queue.write(str.lower()+"\n")
        read_queue.close()

def cleanup(str):
    for i in clean3: str=str.replace(i," ")
    for i in clean2: str=str.replace(i,".")
    for i in clean: str=str.replace(i," ")
    return str

#///////////////////////////////////////////////////////////////

dirlist=os.listdir(filepath)
sze=len(dirlist)
cnt=start
#open up all files
try:
    for txtpath in dirlist[start:]:
        word_cnt=0;word_tot=0;nospace=False
        txt=filepath+"\\"+txtpath
        #sys.stdout.write(f'{Fore.CYAN}{txt}...'+Style.RESET_ALL)
        #prCyan(txt+"...")
        prCyan(f"PROG {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}...")
        #iterate over each word
        start_time=time.time()
        with open(filepath+"\\"+txtpath, 'r', encoding="utf-8") as f:
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
                te=cleanup(te)
                
                #NOTE:check for sentences
                #print(f"<{te}>")
                if "." in te:
                    #print(te.split("."))
                    for i in te.split(".")[:-1]:
                        words = [x for x in i.split(" ") if x]#remove empty spaces
                        #print(words)
                        for wrd in words:
                            addword(wrd)
                    te=te.split(".")[-1]
                
                #input("A")
        nowtime=time.time()
        prYellow( f"{  goodtime(nowtime-start_time)  }\t+{word_cnt}/{word_tot} <{gdFL(100*word_cnt/word_tot)}%> words\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
        t_str=f"PROG {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}..."
        logger(printpath+'gutenburg_log.txt',   f"{t_str}{'.'*(55-len(t_str))}\t{  goodtime(nowtime-start_time)  }\t+{word_cnt}/{word_tot} <{gdFL(100*word_cnt/word_tot)}%> words\t<{   goodtime(nowtime-script_time)   }> RUNTIME\t{last_word}")
        cnt+=1
        #input("B")
        #break
except Exception as e:
    nowtime=time.time()
    logger(printpath+'gutenburg_log.txt',   f"FAILLLLLLL PROG {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}...\t{  goodtime(nowtime-start_time)  }\t+{word_cnt}/{word_tot} <{gdFL(100*word_cnt/word_tot)}%> words\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
    prALERT(f"te:\t\t{te}")
    prALERT(f"words:\t\t{words}")
    prALERT(f"wrd:\t\t{wrd}")
    prALERT(e)

#NOTE: write temp over to csv
with open(printpath+"gutenburg.csv", 'w', encoding="utf-8") as csv_f:
    csv_f.write("word,acceptance\n")
    f=open(printpath+'gutenburg.txt','r', encoding="utf-8")
    #reads each line at a time instead of usual array iteration method, less memory required
    while True:
        t_read= f.readline()
        if t_read == "": break
        csv_f.write(t_read)
    f.close()

#NOTE: remove temp
#os.remove("temp_dict.txt")
