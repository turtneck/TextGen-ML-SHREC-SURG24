import os,sys,math
import matplotlib.pyplot as plt
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.abspath("")
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path)
from fun_colors import *


#================================================================
def DICT_graph(path,startline=1,endclip=None,graphstart=0,graphend=None):
    with open(path, 'r', encoding="utf-8") as f: data = f.readlines()

    if endclip: data=data[startline:-endclip]
    else: data=data[startline:]
    data2=[]
    for i in data: data2.append(i.split('\t'))
    data=data2;del data2

    cnt=0;step=0;last_x=0
    #----
    arr1=[];arr2=[]
    for i in data:
        #================
        #x: #tokens: 256*(cnt+1)*(step+1)*24
        #y:loss
        #================
        #cnt
        if i[0][:4] == 'PROG':
            # cnt = int( i[0].split(" ")[1].split("/")[0] )
            step=0
        if i[0][:4] != 'step': continue
        
        step = int( i[0].split(':')[0].split(" ")[1] )+1-step
        x=(64*256*step)+last_x
        
        if i[0].split(" ")[-1] == 'nan': continue
        y=float( i[0].split(" ")[-1] )
                
        last_x=x
        arr1.append(x)
        arr2.append(y)
        
    if graphend:
        arr1=arr1[graphstart:-graphend]
        arr2=arr2[graphstart:-graphend]
    else:
        arr1=arr1[graphstart:]
        arr2=arr2[graphstart:]
    return arr1,arr2

def comb_plot(arr1,arr2):
    addon=max(arr1)
    arr3=[]
    for i in arr2:
        arr3.append(i+addon)
    return arr3

def check_sorted(arr,num):
    t=arr[:]
    t.sort()
    if arr != t:
        prALERT(f"ERROR: PLOT{num}")
        prRed(arr); prRed(t)


plotX1,plotY1 = DICT_graph(getDrive()+"PT-ChatBot-Type2_1.txt",startline=4)
check_sorted(plotX1,'1')
#----
plotX2,plotY2 = DICT_graph(getDrive()+"PT-ChatBot-Type2_2.txt",startline=4)
check_sorted(plotX2,'2_1')
plotX2 = comb_plot(plotX1,plotX2)
check_sorted(plotX2,'2_2')
Lx2=plotX2[0]
#----
plotX3,plotY3 = DICT_graph(getDrive()+"PT-ChatBot-Type2_3.txt",startline=4)
check_sorted(plotX2,'3_1')
plotX3 = comb_plot(plotX2,plotX3)
check_sorted(plotX2,'3_2')
Lx3=plotX3[0]


plotX = plotX1[1:]+plotX2[1:]+plotX3[1:]
plotY = plotY1[1:]+plotY2[1:]+plotY3[1:]
del plotX1,plotX2,plotX3,plotY1,plotY2,plotY3

min_x=min(plotX);min_y=min(plotY)
max_x=max(plotX);max_y=max(plotY)
prCyan(f'min_x,min_y:\t{min_x}, {min_y}')
prCyan(f'max_x,max_y:\t{max_x}, {max_y}')
# ===========================================

python_str_error='PT-ChatBot-Type2'
plt.xlabel('Tokens Processed')
plt.ylabel('training loss')

xmin, xmax, ymin, ymax = plt.axis()
# plt.axis(xmin=min_x-(xmax/20),ymin=min_y-((ymax-min_y)/20))

plt.plot(plotX,plotY,color='red' )
plt.axvline(x = Lx2, color = 'b')
plt.axvline(x = Lx3, color = 'b')

plt.title(python_str_error)
# plt.savefig(getDrive()+f'Graphs\Graphs\{python_str_error}__{datestr()}.png')
plt.show()