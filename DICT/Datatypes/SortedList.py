# Implementing Red-Black Tree in Python


import sys,tiktoken,random,os,shutil,pickle
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path[:-15])
from fun_colors import *


#-------------------------

class SortList():
    def __init__(self):
        self.arr = []
        self.size=0

    def insert(self, key):
        if not key in self.arr:
            self.size+=1
            index=0 #least-max list
            for i in self.arr:
                if key<i:break
                index+=1
            self.arr.insert(index,key)
        return
    
sl=SortList()
import random
inlist=list('abcdef')
random.shuffle(inlist)
for i in inlist: sl.insert(i)
print(sl.arr)