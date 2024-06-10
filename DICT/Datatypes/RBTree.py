




# Implementing Red-Black Tree in Python


import sys,tiktoken,random,os,shutil,pickle
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path[:-15])
from fun_colors import *

# Node creation
class Node():
    def __init__(self, item, value=0):
        self.item = item
        self.parent = None
        self.left = None
        self.right = None
        self.red = True
        self.value=value


class RBT():
    def __init__(self, file=None):
        self.root = None
        self.arr=None
        self.str=None
        self.size=0
        self.f = None
        
        if file:# and os.path.getsize(file) > 0:
            self.f=file[:-4]+"_t.txt"
            file_wipe(self.f)
            if os.path.getsize(file) > 0:
                #TODO: PROPER PLACEMENT OF ARRAY
                # with open(file, 'rb') as f: arr = pickle.load(f)
                arr = sorted_byVAL(file)
                # random.shuffle(arr)
                # for wrd in arr:
                #     # self.insert(wrd[0],wrd[1])
                #     self.insert(wrd)
                self.balanced_arr_insert_helper(arr)
            else: print("RBT: File doesn't exist, init'd empty")
        else: print("RBT: Wasn't given file, init'd empty")
        print(file)
    
    def balanced_arr_insert_helper(self, arr):
        if not arr: return None
        self.insert( arr[int( (len(arr))/2 )] )
        self.insert( self.balanced_arr_insert_helper( arr[:int( (len(arr))/2 )] ) )
        self.insert( self.balanced_arr_insert_helper( arr[int( (len(arr))/2 )+1:] ) )
                

    # Search the tree
    def search_tree_helper(self, node, key):
        if not node or key == node.item: return node
        if key < node.item: return self.search_tree_helper(node.left, key)
        return self.search_tree_helper(node.right, key)

    def searchTree(self, k):
        return self.search_tree_helper(self.root, k)

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left:
            y.left.parent = x

        y.parent = x.parent
        if x.parent == None: self.root = y
        elif x == x.parent.left: x.parent.left = y
        else: x.parent.right = y
        y.left = x
        x.parent = y

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right: y.right.parent = x

        y.parent = x.parent
        if x.parent == None: self.root = y
        elif x == x.parent.right: x.parent.right = y
        else: x.parent.left = y
        y.right = x
        x.parent = y

    def insert(self, key,value=0):
        if key is None: return
        node = Node(key,value)

        y = None; x = self.root
        while x:
            y = x
            if node.item == x.item: x.value = x.value+1; return
            if node.item < x.item: x = x.left
            else: x = x.right

        self.size+=1
        if self.f: logger(self.f, key)
        node.parent = y
        if y == None: self.root = node
        elif node.item < y.item: y.left = node
        else: y.right = node
        
        if node.parent == None: node.red = False; return
        if node.parent.parent == None: return

        self.fix_insert(node)

    # Balance the tree after insertion
    def fix_insert(self, k):
        while k.parent.red:
            if k.parent == k.parent.parent.right and k.parent.parent.left:
                if k.parent.parent.left.red:
                    k.parent.parent.left.red = False
                    k.parent.red = False
                    k.parent.parent.red = True
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self.right_rotate(k)
                    k.parent.red = False
                    k.parent.parent.red = True
                    self.left_rotate(k.parent.parent)
            else:
                if k.parent.parent.right and k.parent.parent.right.red:
                        k.parent.parent.right.red = False
                        k.parent.red = False
                        k.parent.parent.red = True
                        k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self.left_rotate(k)
                    k.parent.red = False
                    k.parent.parent.red = True
                    self.right_rotate(k.parent.parent)
            if k == self.root: break
        self.root.red = False
        
    def inorder_arr(self):
        self.arr=[]
        self.inorder_arr_helper(self.root)
        arr_c=self.arr.copy()
        self.arr=None
        return arr_c
         
    def inorder_arr_helper(self, node):
        if node:
            self.inorder_arr_helper(node.left)
            self.arr.append(node.item)
            self.inorder_arr_helper(node.right)
        
    def inorder_arr_VAL(self):
        self.arr=[]
        self.inorder_arr_helper_VAL(self.root)
        arr_c=self.arr.copy()
        self.arr=None
        return arr_c
         
    def inorder_arr_helper_VAL(self, node):
        if node:
            self.inorder_arr_helper_VAL(node.left)
            self.arr.append([node.item,node.value])
            self.inorder_arr_helper_VAL(node.right)
        
    def inorder_str(self):
        self.str=''
        self.inorder_str_helper(self.root)
        str_c=self.str[:]
        self.str=None
        return str_c[:-1]
         
    def inorder_str_helper(self, node):
        if node:
            self.inorder_str_helper(node.left)
            self.str += node.item +' '
            self.inorder_str_helper(node.right)

    def print_tree(self):
        self.__print_helper(self.root, "", True)

    # Printing the tree
    def __print_helper(self, node, indent, last):
        if node:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R<")
                indent += " "
            else:
                sys.stdout.write("L<")
                indent += "| "

            s_color = "RED" if node.red else "BLACK"
            print(str(node.item) + ">(" + s_color + ")")
            self.__print_helper(node.left, indent, False)
            self.__print_helper(node.right, indent, True)
        
    def save_tree(self,file):
        # encode = tiktoken.get_encoding("gpt2").encode(self.inorder_str())
        # encode_ids = np.array(encode, dtype=np.uint16)
        # encode_ids.tofile(file)
        # print(self.f)
        # print(file)
        with open(os.path.join(os.path.dirname(__file__), file), 'wb') as f:
            pickle.dump(self.inorder_arr_VAL(), f)
    
        

    



if __name__ == "__main__":
    bst = RBT()

    for _ in range(7): bst.insert(random.randint(0,100))
    bst.print_tree()
    # print(bst.inorder_arr())
    # print(bst.inorder_arr_VAL())
        
    
    # # # print(bst.inorder_arr())
    # # # print(bst.inorder_str())
    
    # # #tried compressing to int, but was bigger as that
    # # import os
    # # dir_path = os.path.dirname(os.path.realpath(__file__))
    # # # bst = RBT(os.path.join(dir_path, 'RBT.bin'))
    # # #bst.print_tree()
    
    # bst.save_tree(os.path.join(dir_path, 'RBT2.bin'))
    
    # bst2 = RBT(os.path.join(dir_path, 'RBT2.bin'))
    # bst2.print_tree()
    # print(bst2.inorder_arr())
    # # print(bst2.inorder_str())
    # # # print(bst.size)
    # # print(bst2.size)
    
    # bst2.save_tree(os.path.join(dir_path, 'RBT3.bin'))
    
    # bst3 = RBT(os.path.join(dir_path, 'RBT3.bin'))
    # bst3.print_tree()
    # print(bst3.inorder_arr())
    # print(bst3.inorder_arr_VAL())
    # import datetime
    # dstr=f"{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}"
    # printpath=(getDrive()+"book/gutenDICT-RBT/char/")
    # # bst.save_tree(printpath+f'gutenDICT-RBT/gutenburg_dict-RBT-chr_FAIL__{dstr}.bin')
    # # bst.save_tree(printpath+f'gutenDICT-RBT/gutenburg_dict-RBT-chr__{dstr}.bin')
    # file="gutenburg_dict-RBT-chr"
    # bst2= RBT( printpath+file+'.bin' )
    # # print(bst2.inorder_arr_VAL())
    
    # # type_dict="word"
    # # file="gutenburg_dict-RBT-word__2024-05-21_6_39"
    # # print( sorted_RBT(printpath+type_dict+'/'+file+'.bin')[:100] )
    # import math
    # print( -1*math.log(1))
    # print( -1*math.log(208449))
    # print(sorted_byVAL(printpath+file+'.bin'))
    
    data = np.memmap("D:/book/gutenburg_BIN/char/GB_pg1.bin",dtype=np.uint16, mode='r')
    print(data[0])