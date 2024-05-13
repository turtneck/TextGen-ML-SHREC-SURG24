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
import csv,os,sys,time
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))[:-5]
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path)
from fun_colors import *

filepath = getDrive()+"book\\gutenburg"
print(f"DRIVE_DIR:\t\t<{filepath}>")


#///////////////////////////////////////////////////////////////
#NOTE: manuals
clean=[',','--','---','[',']',';','*','™','•',':','"',"'",'“','”','(',')','&','=','�','‘','’']#remove
clean2=['***','?','!']#replace with '.'s


#///////////////////////////////////////////////////////////////
def addword(str):
    #NOTE: check if word in file
    inside=False
    with open('temp_dict.txt','r') as read_queue:
        #reads each line at a time instead of usual array iteration method, less memory required
        while True:
            t_read= read_queue.readline()
            if t_read == "": break
            if "\n" in t_read:line=t_read[:-1]
            else: line=t_read
            if str.lower() == line: inside=True            
    #NOTE: if not inside, add to file
    if not inside:
        open('temp_dict.txt','a', encoding="utf-8").write(str.lower()+"\n")

def cleanup(str):
    for i in clean2: str=str.replace(i,".")
    for i in clean: str=str.replace(i,"")
    return str

#///////////////////////////////////////////////////////////////
printpath=filepath.split("\\")[0]+"\\"+filepath.split("\\")[1]+"\\"
tf = open("temp_dict.txt","w");tf.close()#create/clear temp

dirlist=os.listdir(filepath)
#open up all files
for txtpath in dirlist:
    txt=filepath+"\\"+txtpath
    sys.stdout.write(f'{Fore.CYAN}{txt}...'+Style.RESET_ALL)
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
            if line == "</pre></body></html>": break
            #if "*** END OF THE PROJECT GUTENBERG" in line: break
            #NOTE:cleanup
            if not line or line.isspace(): continue
            if "\n" in line:te+=" "+line[:-1]
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
    prYellow( f"\t{gdFL(time.time()-start_time)}s" )
    #input("B")
    #break
        
#NOTE: write temp over to csv
with open(printpath+"gutenburg.csv", 'w') as csv_f:
    csv_f.write("word,acceptance\n")
    f=open('temp_dict.txt','r')
    #reads each line at a time instead of usual array iteration method, less memory required
    while True:
        t_read= f.readline()
        if t_read == "": break
        csv_f.write(t_read)
    f.close()

#NOTE: remove temp
os.remove("temp_dict.txt")
