import csv,os,sys,time,numpy,datetime,multiprocessing,re
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path[:-14])
from fun_colors import *
from PT_Container_v1 import PT_model_v1, PTV1_HYPER_DEF
from PT_Container_v2 import PT_model_v2_1, PTV2_1_HYPER_DEF



print( os.listdir(dir_path+'\Models'))

for i in os.listdir(dir_path+'\Models'):
    if i[3] == '1':
        print('1',i[3])
        MODEL = PT_model_v1(
        meta_data=getDrive()+"book/gutenburg_BIN/metas/gutenburg_bin-RBT-char_meta_int64.pkl",
        model_path=dir_path+'\Models\\'+i
        )
    elif i[3] == '2':
        print('2',i[3])
        MODEL = PT_model_v2_1(
        meta_data=getDrive()+"book/gutenburg_BIN/metas/gutenburg_bin-RBT-char_meta_int64.pkl",
        model_path=dir_path+'\Models\\'+i
        )
    else: prALERT('FAIL:',i[3])
    prGreen(i)
    print( MODEL.run_model() )