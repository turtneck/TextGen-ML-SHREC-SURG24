import os,sys,math
import matplotlib.pyplot as plt
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.abspath("")
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path)
from fun_colors import *


#================================================================
def DICT_graph2(path,startline=5,endclip=None):
    with open(path, 'r', encoding="utf-8") as f: data = f.readlines()

    if endclip: data=data[startline:-endclip]
    else: data=data[startline:]
    data2=[]
    for i in data: data2.append(i.split('\t'))
    data=data2;del data2

    cnt=0;step=0;last_x=0
    arr1=[];arr2=[]
    #----
    arr1_t=[];arr2_t=[]
    for i in data:
        # print(i)
        #================
        #x: #tokens: 256*(cnt+1)*(step+1)*24
        #y:loss
        #================
        #cnt
        if i[0][:4] == 'PROG':
            # print(arr1[:10])
            if arr1 and arr2:
                # prCyan("hi")
                arr1_t.append(arr1)
                arr2_t.append(arr2)
                arr1=[];arr2=[]
                last_x=0
            step=0
        if i[0][:4] != 'step': continue
        
        #step
        step = int( i[0].split(':')[0].split(" ")[1] )+1-step
        #------------
        
        x=(24*256*step)+last_x
        y=float( i[0].split(" ")[-1] )
        
        last_x=x
        arr1.append(x)
        arr2.append(y)
    return arr1_t,arr2_t

path = getDrive()+"PTv2_CRC-pretrain2_2024-07-26_3_20.txt"
plotX_t,plotY_t = DICT_graph2(path)
prCyan("DICT_graph2")

cnt=0;length=len(plotX_t)
for i in range(len(plotX_t)):
    plotX=plotX_t[i]
    plotY=plotY_t[i]
    a=1-( (cnt+5)/(length+5) )
    plt.plot(plotX,plotY,color=(1,a,a) )
    cnt+=1
prCyan("plotted")
# ===========================================

python_str_error='PTv2 PreTraining'
plt.xlabel('Tokens Processed')
plt.ylabel('training loss')

# xmin, xmax, ymin, ymax = plt.axis()
# plt.axis(xmin=min_x-(xmax/20),ymin=min_y-((ymax-min_y)/20))


plt.title(python_str_error)
prCyan("ready")
plt.show()