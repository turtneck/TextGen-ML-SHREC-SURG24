# take a log file and turn it into a graphable list for excel


import os,sys,math
import matplotlib.pyplot as plt
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path[:-9])
from fun_colors import *



#------------------------------
def DICT_graph(path,startline=1,endclip=None,graphstart=0,graphend=None):
    with open(path, 'r', encoding="utf-8") as f: data = f.readlines()

    if endclip: data=data[startline:-endclip]
    else: data=data[startline:]
    data2=[]
    for i in data: data2.append(i.split('\t'))
    data=data2;del data2
    #----
    arr1=[];arr2=[]
    for i in data:
        #x:step, y:loss
        
        x=int( i[0].split(" ")[1][:-1] )
        y=float( i[0].split(" ")[-1] )
        
        arr1.append(x)
        arr2.append(y)
        
    if graphend:
        arr1=arr1[graphstart:-graphend]
        arr2=arr2[graphstart:-graphend]
    else:
        arr1=arr1[graphstart:]
        arr2=arr2[graphstart:]
    return arr1,arr2

# ===========================================
path=getDrive()+'Graphs/Data/PTv2_Batch/'

#---
avg_arr=[]
avg_max=math.inf
min_x=math.inf
min_y=math.inf
cnt=0

dirlist=os.listdir(path)
for filepath in dirlist:
    plotX,plotY = DICT_graph(path+filepath,startline=4)
    if avg_max>mean(plotY):
        avg_max=mean(plotY)
        avg_arr = [cnt, plotX, plotY]
    if min_x>min(plotX):min_x=min(plotX)
    if min_y>min(plotY):min_y=min(plotY)
    a=1-( (cnt+5)/(len(dirlist)+5) )
    plt.plot(plotX,plotY,color=(1,a,a) )
    cnt+=1


# ===========================================
prCyan(f'min_x,min_y:\t{min_x}, {min_y}')
python_str_error='PTv2 BatchSizes'
plt.xlabel('steps')
plt.ylabel('training loss')

xmin, xmax, ymin, ymax = plt.axis()
plt.text(xmax/10,(4.5*ymax)/5,'done')
plt.plot(avg_arr[1],avg_arr[2],color='blue' ) #replot best average's line
plt.axis(xmin=min_x-(xmax/20),ymin=min_y-((ymax-min_y)/20))

plt.title(python_str_error+f': { avg_arr[0] }, { gdFL(avg_max) }avg')
plt.savefig(getDrive()+f'Graphs\Graphs\{python_str_error}__{datestr()}.png')
plt.show()