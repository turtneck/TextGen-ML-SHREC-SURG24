
import os,re
import numpy as np

global dir_path
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t{dir_path}>")

def red(filename):
    arr=[]
    with open(dir_path+"/"+filename, 'r') as f:
        line = f.readline()
        for i in line.split(" "):
            if i not in [" ","\n"]:
                arr.append(i)
    return arr


print( red("init_news/news0.txt") )