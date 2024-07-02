# holder file for easier printing of colored terminal text for readibility
# taken from: https://www.geeksforgeeks.org/print-colors-python-terminal/

import pickle,datetime,os
# from colorama import Fore, Back, Style
def prRed(skk): print("\033[91m{}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m{}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m{}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m{}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m{}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m{}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m{}\033[00m" .format(skk))
def prBlack(skk): print("\033[98m{}\033[00m" .format(skk))


#mine
def prALERT(skk):print(Back.RED+skk+Style.RESET_ALL)
def gdFL(fl): return f"{'{:.2f}'.format(float( fl ))}"

def logger(filepath,text):
    file = open(filepath,'a', encoding="utf-8")
    file.write(text+"\n")
    file.close()

def getDrive():
    # return str(os.path.abspath(""))+'//'
    # return 'SURG24//'
    return ''

def goodtime(tim):
    str=""
    if tim>86400: str+=f"{int(tim/86400)}d "; tim=tim%86400
    if tim>3600: str+=f"{int(tim/3600)}h "; tim=tim%3600
    if tim>60: str+=f"{int(tim/60)}m "; tim=tim%60
    str+= gdFL(tim)+"s"
    return str

#if file doesnt exist, make it. otherwise dont effect it
def file_helper(path):
    t_file= open(path,'a', encoding="utf-8")
    t_file.close()

def file_wipe(path):
    t_file= open(path,'w', encoding="utf-8")
    t_file.close()

def fun_encode(s,stoi):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def fun_decode(l,itos):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

def load_RBT_Arr(file):
    print(file)
    with open(file, 'rb') as f: dat = pickle.load(f)
    return dat
        
def sublist_sort(list_tb: list): #sort list of lists by 2nd element
    list_tb.sort(key = lambda x: x[1], reverse=True)
    return list_tb

def sorted_RBT(file):
    return sublist_sort(load_RBT_Arr(file))

def sorted_byVAL(file):
    return [item[0] for item in sublist_sort(load_RBT_Arr(file))]

def rgb(minimum, maximum, value):
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = max(0, (1 - ratio))
    r = max(0, (ratio - 1))
    g = 1-b-r
    return (r, g, b)

def datestr():
    return f'{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}'

def mean(arr):
    tot=0
    for i in arr:tot+=i
    return tot/len(arr)