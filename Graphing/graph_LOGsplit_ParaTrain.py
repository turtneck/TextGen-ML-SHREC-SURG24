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
startline = 4
endclip = 2
graphstart=6
graphend=0


#------------------------------
inp1=input("Directory or Specific file? (1/2):\t")
inp2= input("PATH:\t")

#------------------------------
# specific log file
if inp1 == "1": dirlist=os.listdir(inp2) # whole directory
elif inp1 == "2": dirlist=[inp2] # specific log file
else: raise NameError("WRONG SELECTION")
avg_arr=[]
avg_max=0
min_x=math.inf

#read through data
cnt=0
for txtpath in dirlist:
    # Generic
    if inp1 == "1":
        with open(inp2+'\\'+txtpath, 'r', encoding="utf-8") as f: data = f.readlines()
    elif inp1 == "2":
        with open(txtpath, 'r', encoding="utf-8") as f: data = f.readlines()
    if endclip: data=data[startline:-endclip]
    else: data=data[startline:]
    data2=[]
    for i in data: data2.append(i.split('\t'))
    data=data2;del data2
    # for i in data: print(i)
    
    #------------
    # needs to be adjusted manually
    arr1=[];arr2=[]
    for i in data:
        a=i[0].split(" ")
        x = a[1][:-1]
        b=i[1].split(" ")
        y=0
        if len(b)>=1: y+= float( b[-1][:-1] )
        if len(b)>=2: y+= float( b[-2][:-1] )*60
        if len(b)>=3: y+= float( b[-3][:-1] )*3600
        if len(b)>=4: y+= float( b[-4][:-1] )*86400
        # print( f"{gdFL(x)},\t{gdFL(y)},\t{gdFL(float(a[-1]))},\t{gdFL(y/float(a[-1])**2)},\t{gdFL(1000/(y*float(a[-1])**2))},\t{gdFL(1000/(y*float(a[-1])))},\t{gdFL(100000/(y*(float(a[-1])**4)))}" )
        # print( f"{gdFL(x)},\t{gdFL(100000/(y*(float(a[-1])**4)))}" )
        arr1.append(int(x))
        arr2.append(round(100000/(y*(float(a[-1])**4)),2))
        
    if graphend:
        arr1=arr1[graphstart:-graphend]
        arr2=arr2[graphstart:-graphend]
    else:
        arr1=arr1[graphstart:]
        arr2=arr2[graphstart:]
    if avg_max<mean(arr2):
        avg_max=mean(arr2)
        avg_arr = [cnt, arr1, arr2]
    if min_x>min(arr1):min_x=min(arr1)
    # print( rgb(0,len(dirlist)-1,cnt) )
    # plt.plot(arr1,arr2,color=rgb(0,len(dirlist)-1,cnt) )
    a=1-( (cnt+5)/(len(dirlist)+5) )
    plt.plot(arr1,arr2,color=(1,a,a) )
    cnt+=1


python_str_error = inp2.split('\\')[-1]
plt.xlabel('steps')
plt.ylabel('training loss effectiveness')
xmin, xmax, ymin, ymax = plt.axis()
plt.text(xmax/10,(4.5*ymax)/5,'eff= 100k/( loss^4 * time )')
plt.plot(avg_arr[1],avg_arr[2],color='blue' ) #replot best average's line
plt.axis(xmin=min_x)
plt.title(python_str_error+f': { avg_arr[0] }, { gdFL(avg_max) }avg')
plt.savefig(getDrive()+f'Graphs\Graphs\{python_str_error}__{datestr()}.png')
plt.show()