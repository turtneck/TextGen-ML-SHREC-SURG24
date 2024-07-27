'''
BOILERPLATE:

Reads from a bunch of .txt files in a directory and adds characters to a Red-Black Tree, adding the frequency each character appears.
Made specifically for the Gutenburg dataset.

This method is RAM heavy, and 100000x faster then the 'minRAM' approach that tries to make RAM use a little as possible
'''
#///////////////////////////////////////////////////////////////
#imports
import csv,os,sys,time,numpy,datetime,multiprocessing,re
import numpy as np
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.abspath("")
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path)
from fun_colors import *

#file path and global variables
script_time=time.time()
print(f"DRIVE_DIR:\t\t<{getDrive()+'book/gutenburg'}>")
printpath=(getDrive()+"book/dict/")
log_name=printpath+'gutenburg_log-RBT-chr-TEST.txt'
print(f"LOG_DIR:\t\t<{log_name}>")
file_helper( log_name )#if log doesnt exist make it
logger(log_name,   f"\n\n[!!!!!] START\t{str(datetime.datetime.now())}")


#loading past dict---------------------
dict_name=printpath+'gutenburg_dict-RBT-chr-TEST.bin'
#///////////////////////////////////////////////////////////////



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



#///////////////////////////////////////////////////////////////
print(f"DICT_FILE:\t\t<{ dict_name }>")
file_helper( dict_name )#if dict doesnt exist make it
dict = RBT( dict_name )

dstr=f"{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}"
fail=False
#///////////////////////////////////////////////////////////////
#NOTE: manuals

start=0

#------------------------
dirlist=os.listdir(getDrive()+"book/gutenburg")
sze=len(dirlist)-1
cnt=start
#open up all files
try:
    for txtpath in dirlist[start:]:
        last_word="";nospace=False
        txt=getDrive()+"book/gutenburg/"+txtpath
        prCyan(f"PROG {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}...")
        
        
        #load whole data set into RAM (one big string) and format it down to words
        start_time=time.time()
        with open(getDrive()+"book/gutenburg/"+txtpath, 'r', encoding="utf-8") as f: data = f.readlines()[1:-1]
        data = ''.join(data)
        data=data_clean(data)
        word_cnt = dict.size
        for chr in data: dict.insert(chr)
        
        word_cnt=dict.size-word_cnt
        nowtime=time.time()
        prYellow( f"{  goodtime(nowtime-start_time)  }\t+<{word_cnt}> chars\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
        t_str=f"PROG {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>  {txt}..."
        logger(log_name,   f"{t_str}{'.'*(56-len(t_str))}\t+<{word_cnt}> chars\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME\t{last_word}")
        cnt+=1
        #------------------
except Exception as e:
    fail=True
    nowtime=time.time()
    logger(log_name,   f"FAILLLLLLL PROG<> {cnt}/{sze}: <{gdFL( 100*cnt/sze )}%>\t{txt}...\t{  goodtime(nowtime-start_time)  }\t<{   goodtime(nowtime-script_time)   }> RUNTIME")
    prALERT(f"dict size:\t\t{dict.size}")
    prALERT(e)
    
    

#///////////////////////////////////////////////////////////////
if not fail: print("bro passed")
else: print(f"bro failed:\t{cnt}\t{txt}")