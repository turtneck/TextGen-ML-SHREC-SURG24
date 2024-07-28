# take a log file and turn it into a graphable list for excel


import os,sys,math
import matplotlib.pyplot as plt
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path[:-9])
from fun_colors import *
#------------------------------
# manual
startline = 1
endclip = None
graphstart=0
graphend=None


#------------------------------
def DICT_graph(path,startline=1,endclip=None,graphstart=0,graphend=None):
    with open(path, 'r', encoding="utf-8") as f: data = f.readlines()

    if endclip: data=data[startline:-endclip]
    else: data=data[startline:]
    data2=[]
    for i in data: data2.append(i.split('\t'))
    data=data2;del data2
    #----
    lasty=0
    arr1=[];arr2=[]
    for i in data:      
        #x:cnt, y:time
        x=int( i[0].split(" ")[1].split("/")[0] )
        #--
        b=i[3][1:-9].split(" ")
        y=0
        for j in b:
            if j[-1]=='s': y+= float( j[:-1] )
            if j[-1]=='m': y+= float( j[:-1] )*60
            if j[-1]=='h': y+= float( j[:-1] )*3600
            if j[-1]=='d': y+= float( j[:-1] )*86400
        #--
        # print(f'x:{x},\ty:{y}')
        if y<lasty: prCyan(f'x:{x},\ty:{y}\t{i}')
        lasty=y
        
        arr1.append(x)
        arr2.append(y)
        
    if graphend:
        arr1=arr1[graphstart:-graphend]
        arr2=arr2[graphstart:-graphend]
    else:
        arr1=arr1[graphstart:]
        arr2=arr2[graphstart:]
    return arr1,arr2

#------------------------------
# specific log file

path=getDrive()+'/Graphs/Data/'

# MaxHeap_X,MaxHeap_Y = DICT_graph(path+'gutenburg_log-MaxHeap-chr-TEST.txt')
# plt.plot(MaxHeap_X,MaxHeap_Y,color='red' )
# MinHeap_X,MinHeap_Y = DICT_graph(path+'gutenburg_log-MinHeap-chr-TEST.txt')
# plt.plot(MinHeap_X,MinHeap_Y,color='orange' )
# numpy_X,numpy_Y     = DICT_graph(path+'gutenburg_log-numpyUnsortList-chr-TEST.txt')
# plt.plot(numpy_X,numpy_Y,color='yellow' )
# RBTree_X,RBTree_Y   = DICT_graph(path+'gutenburg_log-RBT-chr-TEST.txt')
# plt.plot(RBTree_X,RBTree_Y,color='green' )
# SList_X,SList_Y     = DICT_graph(path+'gutenburg_log-SortList-chr-TEST.txt')
# plt.plot(SList_X,SList_Y,color='cyan' )
# UsList_X,UsList_Y   = DICT_graph(path+'gutenburg_log-UnsortList-chr-TEST.txt')
# plt.plot(UsList_X,UsList_Y,color='blue' )
#---
plotX,plotY = DICT_graph(path+'gutenburg_log-MaxHeap-chr-TEST.txt')
plt.plot(plotX,plotY,color='red' )
plotX,plotY = DICT_graph(path+'gutenburg_log-MinHeap-chr-TEST.txt')
plt.plot(plotX,plotY,color='orange' )
plotX,plotY     = DICT_graph(path+'gutenburg_log-numpyUnsortList-chr-TEST.txt')
plt.plot(plotX,plotY,color='yellow' )
plotX,plotY   = DICT_graph(path+'gutenburg_log-RBT-chr-TEST.txt')
plt.plot(plotX,plotY,color='green' )
plotX,plotY     = DICT_graph(path+'gutenburg_log-SortList-chr-TEST.txt')
plt.plot(plotX,plotY,color='blue' )
plotX,plotY   = DICT_graph(path+'gutenburg_log-UnsortList-chr-TEST.txt')
plt.plot(plotX,plotY,color='purple' )




python_str_error='DICTs'
plt.xlabel('Dataset Count')
plt.ylabel('Process Time')
# xmin, xmax, ymin, ymax = plt.axis()
# plt.text(xmax/10,(4.5*ymax)/5,'eff= 100k/( loss^4 * time )')
# plt.plot(avg_arr[1],avg_arr[2],color='blue' ) #replot best average's line
plt.axis(xmin=0,ymin=0)
plt.title( 'Adding unique tokens to DataStructs (Over 13 hours)' )
plt.savefig(getDrive()+f'Graphs\Graphs\{python_str_error}__{datestr()}.png')
plt.show()