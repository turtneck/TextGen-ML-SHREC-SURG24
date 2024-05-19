
print(   "a">"st"   )
print(   "a"<"st"   )
print(   "a">"s"   )

#print( int("a"), int("st") )




import os, requests, tiktoken
import numpy as np
enc = tiktoken.get_encoding("gpt2")

#train_ids = enc.encode_ordinary(train_data)


#///////////////////////////////////////////////////////////////
#imports
import csv,os,sys,time,numpy,datetime,multiprocessing
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path[:-15])
from fun_colors import *

#file path and global variables
#///////////////////////////////////////////////////////////////
print(f"DRIVE_DIR:\t\t<{getDrive()+'book/gutenburg'}>")
dirlist=os.listdir(getDrive()+"book\\gutenburg")
txt=getDrive()+"book\\gutenburg"+"\\"+dirlist[1]
print(txt)

# with open(txt, 'r', encoding='utf-8') as f: data = f.read()

# encode = enc.encode_ordinary(data)
# encode_ids = np.array(encode, dtype=np.uint16)
# print(data[0])
# print(encode[0])
# print(encode_ids[0])
# encode_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.txt'))



#///////////////////////////////////////////////////////////////
# print("=================")
# print(   enc.encode_ordinary("a")   )
# print(   enc.encode_ordinary("b")   )
# print(   enc.encode_ordinary("aa")   )
# print(   enc.encode_ordinary("ab")   )
# print(   enc.encode_ordinary("ba")   )
# print(   enc.encode_ordinary("bb")   )

# print(   enc.decode([64])   )
# print(   enc.decode([65])   )
# print(   enc.decode([7252])   )
# print(   enc.decode([397])   )
# print(   enc.decode([7012])   )
# print(   enc.decode([11848])   )


# print(   enc.decode(enc.encode_ordinary("a"))   )


# #//////////////////////////
# #test curr dict
# with open(txt, 'r', encoding='utf-8') as f: data1 = f.readlines()
# cnt=0;bl=False
# for line in data1:
#     if line != enc.decode(enc.encode_ordinary( line )):
#         print(f"<{line}>, {cnt}");bl=True
#     cnt+=1
# print("bl:",bl)
# # print(data1)


#//////////////////////////
# with open(txt, 'r', encoding='utf-8') as f: data1 = f.read()
# for i in ['™']: data1=data1.replace(i,"")
# for i in [',','--','---','[',']',';','*','•',':','"','“','”','(',')','&','=','�','—','\t','/','\\','_','|','<','>','\n']: data1=data1.replace(i," ")
# for i in ['***','?','!']: data1=data1.replace(i,".")
# for i in ['.\n', '. ']: data1=data1.replace(i," ")
# data1= list(np.unique( data1.split(" ") ))
# data11=[]
# for wrd in data1:
#     if wrd=='':continue
#     if wrd[0] == "'" and wrd[-1] == "'": wrd=wrd[1:-1]
#     elif wrd[0] == "'": wrd=wrd[1:]
#     elif wrd[0] == "‘" and wrd[-1] == "’": wrd=wrd[1:-1]
#     elif wrd[0] == "‘": wrd=wrd[1:]
#     data11.append(wrd)
# cnt=0;bl=False
# for line in data11:
#     if line != enc.decode(enc.encode_ordinary( line )):
#         print(f"<{line}>, {cnt}");bl=True
#     cnt+=1
# # print("bl:",bl)
# # print(data11)




# #///////////////////////////////////////////////////////////////
# print("=================")
# stoi = { ch:i for i,ch in enumerate(data11) }
# itos = { i:ch for i,ch in enumerate(data11) }

# def encode(s):
#     return [stoi[c] for c in s] # encoder: take a string, output a list of integers
# def decode(l):
#     return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# print(   stoi   )

# print( sys.getsizeof("aaaaaaaaaaaaaaaaaaa")  )
# print( sys.getsizeof(enc.encode("aaaaaaaaaaaaaaaaaaa"))  )
# print( sys.getsizeof("a")  )
# print( enc.encode("a")  )
# print( sys.getsizeof(enc.encode("a"))  )
# print( sys.getsizeof(enc.encode("a")[0])  )

# stoi_save = np.array(stoi)
# # print(stoi_save)
# stoi_save.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))





#///////////////////////////////////////////////////////////////
with open(txt, 'r', encoding='utf-8') as f: data1 = f.read()
for i in ['™']: data1=data1.replace(i,"")
for i in [',','--','---','[',']',';','*','•',':','"','“','”','(',')','&','=','�','—','\t','/','\\','_','|','<','>','\n']: data1=data1.replace(i," ")
for i in ['***','?','!']: data1=data1.replace(i,".")
for i in ['.\n', '. ']: data1=data1.replace(i," ")
data1= list(np.unique( data1.split(" ") ))
data11=[]
for wrd in data1:
    if wrd=='':continue
    if wrd[0] == "'" and wrd[-1] == "'": wrd=wrd[1:-1]
    elif wrd[0] == "'": wrd=wrd[1:]
    elif wrd[0] == "‘" and wrd[-1] == "’": wrd=wrd[1:-1]
    elif wrd[0] == "‘": wrd=wrd[1:]
    data11.append(wrd)

data = ' '.join(data11)
# print(data)
print(len(data11))

# encode = enc.encode(data)
# encode_ids = np.array(encode, dtype=np.uint16)
# encode_ids.tofile(os.path.join(os.path.dirname(__file__), 'RBT.bin'))

# decode = np.memmap(os.path.join(dir_path, 'RBT.bin'), dtype=np.uint16, mode='r')
# data2 = enc.decode(decode)
# print(data2)
# print(data == data2)

# print([1,2,3,4,5][:2],[1,2,3,4,5][2:])


# arr= [1,2,3,4,5]
# print([arr[int((len(arr) - 1)/2)]])
# print(arr[:int((len(arr) - 1)/2)])
# print(arr[int((len(arr) - 1)/2)+1:])
# k=[arr[int((len(arr) - 1)/2)]]
# print( np.concatenate(   ([arr[int((len(arr) - 1)/2)]],arr[:int((len(arr) - 1)/2)],arr[int((len(arr) - 1)/2)+1:]), axis=None    ) )


# mid = int((len(arr) - 1)/2)
# print( np.concatenate(   ([arr[mid]],arr[:mid],arr[mid+1:]), axis=None    ) )




with open(txt, 'r', encoding='utf-8') as f: data1 = f.read()
for i in ['™']: data1=data1.replace(i,"")
for i in [',','--','---','[',']',';','*','•',':','"','“','”','(',')','&','=','�','—','\t','/','\\','_','|','<','>','\n']: data1=data1.replace(i," ")
for i in ['***','?','!']: data1=data1.replace(i,".")
for i in ['.\n', '. ']: data1=data1.replace(i," ")
data1 = data1.split(" ")
# indexes = np.unique(data1, return_index=True)[1]
# data1 = [data1[index] for index in sorted(indexes)]

res,ind = np.unique(data1, return_index=True)
data1 = res[np.argsort(ind)]
del res;del ind
data11=[]
for wrd in data1:
    if wrd=='':continue
    if wrd[0] == "'" and wrd[-1] == "'": wrd=wrd[1:-1]
    elif wrd[0] == "'": wrd=wrd[1:]
    elif wrd[0] == "‘" and wrd[-1] == "’": wrd=wrd[1:-1]
    elif wrd[0] == "‘": wrd=wrd[1:]
    data11.append(wrd)
print(len(data11))
print(data11)