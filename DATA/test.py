# #///////////////////////////////////////////////////////////////
# #imports
# import csv,os,sys,time,numpy,datetime
# #sys.path.append('D:/projects/base/app/modules') 
# dir_path = os.path.dirname(os.path.realpath(__file__))[:-5]
# print(f"DIRECTORY:\t\t<{dir_path}>")
# sys.path.append(dir_path)
# from fun_colors import *

# #file path and global variables
# filepath = getDrive()+"book\\gutenburg"
# print(f"DRIVE_DIR:\t\t<{filepath}>")
# printpath=filepath.split("\\")[0]+"\\"+filepath.split("\\")[1]+"\\"
# nospace=False #word-word between a new space consideration
# global word_cnt
# #///////////////////////////////////////////////////////////////

# clean=[',','--','---','[',']',';','*','™','•',':','"','“','”','(',')','&','=','�','—','\t']#remove
# clean2=['***','?','!']#replace with '.'s
# clean3=['_']#replace space

# start=0

# #///////////////////////////////////////////////////////////////
# def cleanup(str):
#     for i in clean3: str=str.replace(i," ")
#     for i in clean2: str=str.replace(i,".")
#     for i in clean: str=str.replace(i," ")
#     return str


# #///////////////////////////////////////////////////////////////


# str='''Importance of Marie Antoinette in the Revolution.--Value of her
# Correspondence as a Means of estimating her Character.--Her Birth,
# November 2d, 1755.--Epigram of Metastasio.--Habits of the Imperial
# Family.--Schönbrunn.--Death of the Emperor.--Projects for the Marriage of
# the Archduchess.--Her Education.--The Abbé de Vermond.--Metastasio.--
# Gluck.'''


# dirlist=os.listdir(filepath)
# sze=len(dirlist)
# cnt=start
# te=""
# for line in str.split("\n"):
#     line=line+"\n"
#     if line == "</pre></body></html>":
#         # words = [x for x in i.split(" ") if x]#remove empty spaces
#         # for wrd in words:
#         #     addword(wrd)
#         break
#     #if "*** END OF THE PROJECT GUTENBERG" in line: break
#     #NOTE:cleanup
#     if not line or line.isspace(): continue
#     #remove starting spaces from line
#     nonspc=0
#     for i in range(len(line)):
#         if line[i] != " ": nonspc=i;break
#     line=line[i:]
#     #print(f"<{line}>")
    
#     if "\n" in line:
#         line=line[:-1]
#         if nospace:
#             te+=line
#             nospace=False
#         elif line[-1] == '-':
#             nospace=True
#             te+=" "+line
#         else: te+=" "+line
#     else:
#         if nospace:
#             te+=line
#             nospace=False
#         elif line[-1] == '-':
#             nospace=True
#             te+=line
#         else: te+=line
#     te=cleanup(te)
    
#     #NOTE:check for sentences
#     #print(f"<{te}>")
#     if "." in te:
#         #print(te.split("."))
#         for i in te.split(".")[:-1]:
#             words = [x for x in i.split(" ") if x]#remove empty spaces
#             #print(words)
#             for wrd in words:
#                 print(wrd)
#         te=te.split(".")[-1]
        
        
        
# print('\\')
# print(len("PROG 517/57588: <0.90%>	D:\book\gutenburg\pq10655.txt..."))



def partition(tot,parts):
    part=tot/parts
    arr=[]
    last=0
    for i in range(parts):
        arr.append([last,  int(part*(i+1)-1)  ])
        last=int(part*i)
    return arr

print( partition(57588,6) )