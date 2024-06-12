# take a log file and turn it into a graphable list for excel


import os,sys
import xlsxwriter as xls
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path[:-9])
from fun_colors import *
#------------------------------
# manual
startline = 4
endclip = 2



#------------------------------
inp1=input("Directory or Specific file? (1/2):\t")
inp2= input("PATH:\t")

#------------------------------
# specific log file
if inp1 == "1": dirlist=os.listdir(inp2) # whole directory
elif inp1 == "2": dirlist=[inp2] # specific log file
else: raise NameError("WRONG SELECTION")
# print(dirlist)

#get max num of lines in dirlist, make array to turn into excel .txt list
max=0
for txtpath in dirlist:
    if inp1 == "1":
        with open(inp2+'\\'+txtpath, 'r', encoding="utf-8") as f: data = f.readlines()[4:-endclip]
    elif inp1 == "2":
        with open(txtpath, 'r', encoding="utf-8") as f: data = f.readlines()[4:-endclip]
    if max<len(data): max=len(data)
    del data
print_arr=['']*(max+1)
str_t=''
for i in range(1,max+1): str_t+=str(i)+'\t\t'
print_arr[0]=str_t;del str_t
# print(print_arr)


#read through data
for txtpath in dirlist:
    # Generic
    if inp1 == "1":
        with open(inp2+'\\'+txtpath, 'r', encoding="utf-8") as f: data = f.readlines()[4:-endclip]
    elif inp1 == "2":
        with open(txtpath, 'r', encoding="utf-8") as f: data = f.readlines()[4:-endclip]
    data2=[]
    for i in data: data2.append(i.split('\t'))
    data=data2;del data2
    # for i in data: print(i)
    
    #------------
    # needs to be adjusted manually
    cnt=0
    for i in data:#[-1:]:
        a=i[0].split(" ")
        # print(a[1][:-1],a[-1])
        # print( float(a[1][:-1])/float(a[-1]) )
        x = a[1][:-1]
        b=i[1].split(" ")
        # print( b )
        y=0
        if len(b)>=1: y+= float( b[-1][:-1] )
        if len(b)>=2: y+= float( b[-2][:-1] )*60
        if len(b)>=3: y+= float( b[-3][:-1] )*3600
        if len(b)>=4: y+= float( b[-4][:-1] )*86400
        # print( x, y, float(a[-1]),  y/float(a[-1])**2,  1/(y*float(a[-1])**2) )
        print( f"{gdFL(x)},\t{gdFL(y)},\t{gdFL(float(a[-1]))},\t{gdFL(y/float(a[-1])**2)},\t{gdFL(1000/(y*float(a[-1])**2))},\t{gdFL(1000/(y*float(a[-1])))},\t{gdFL(100000/(y*(float(a[-1])**4)))}" )
        print_arr[cnt]+=f'{x}\t{gdFL(100000/(y*(float(a[-1])**4)))}\t'
        cnt+=1

# for i in print_arr:print(i)
print(inp2.split('\\')[-1])
python_str_error = inp2.split('\\')[-1]
with open(  getDrive()+f'Graphs\Data\{python_str_error}.txt', 'w', encoding="utf-8") as f:
    for i in print_arr:
        f.write(i+'\n')
    f.close()

import csv
import openpyxl
wb = openpyxl.Workbook()
ws = wb.worksheets[0]

with open(getDrive()+f'Graphs\Data\{python_str_error}.txt', 'r', encoding="utf-8") as data:
    reader = csv.reader(data, delimiter='\t')
    for row in reader:
        ws.append(row)

wb.save(getDrive()+f'Graphs\{python_str_error}.xlsx')
