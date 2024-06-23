'''
BOILERPLATE:

Trainer over all Gutenburg data
'''
#///////////////////////////////////////////////////////////////
#importsimport os,sys,math
import csv,os,sys,time,numpy,datetime,multiprocessing,re
#sys.path.append('D:/projects/base/app/modules') 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"DIRECTORY:\t\t<{dir_path}>")
sys.path.append(dir_path[:-14])
from fun_colors import *
from PT_Container_v1 import PT_model_v1, PTV1_HYPER_DEF



#///////////////////////////////////////////////////////////////
#!! manuals
VERSION = '1'
THREADS = 24


#///////////////////////////////////////////////////////////////
'''
NOTE NOTE NOTE NOTE NOTE NOTE NOTE
This is ripped, needs to be made actually functional
TODO
- needs adjustment to look at all .txt files in '\DATA' directory
- put these directories into an array
    - have directories only have text files
- make 'latest' go through this like 'all known'
- make 'all known's array be these directories
    - start with this version and copy over to 'latest'
'''


#=======================================================
if __name__ == "__main__":
    print(Back.CYAN+"1:Latest  "+Style.RESET_ALL)
    print(Back.CYAN+"2:All Known Data "+Style.RESET_ALL)
    inp0= int(input(Fore.CYAN+"SELECT #: "));print(Style.RESET_ALL)
    
    if inp0 == 1:
        #!! get dir of models
        model_list = os.listdir(getDrive()+f'Models\PyTorch_v{VERSION}\Gutenburg')
        
        #!! load last model
        #   NOTE; each model name is saved in the format:      PTv(version)__(datetime)__(dataset index).pt
        #   this has it sorted by version, time created, dataset ran; each in accending order
        try:
            #find last model
            if len(model_list) == 0: raise IndexError("ERROR:  Model Directory is empty, no 'latest' models to choose from")
            if model_list[-1][-3:] != '.pt': raise IndexError(f"ERROR:  Latest 'Model' is invalid file type: < {model_list[-1][-3:]} >")
            prLightPurple(f'LOADING MODEL:\t<<  {getDrive()+f"Models/PyTorch_v{VERSION}/Gutenburg/"+model_list[-1]}  >>...\nProceed?')
            print(Back.GREEN+"SUCCESS: MODEL FOUND"+Style.RESET_ALL)
            if input('') == 'n': os._exit()
            
            #load model
            MODEL = PT_model_v1(
                meta_data=getDrive()+"book/gutenburg_BIN/metas/gutenburg_bin-RBT-char_meta_int64.pkl",
                model_path=getDrive()+f'Models\PyTorch_v1\Gutenburg/'+model_list[-1]
                )
        except Exception as e:
            prALERT(str(e))
            print(Style.RESET_ALL)
            os._exit()
            
                    
        
        #------------------------
        #!! running
        prRed(Back.RED+f'TRAINING LEN: {len(os.listdir(getDrive()+"book/gutenburg_BIN/char_64")[int( model_list[-1].split("__")[-1][:-3] ):])}'+Style.RESET_ALL)
        input(Fore.CYAN+"Ready to run training? <ENTER>")
        
        MODEL.train_model(
            dir_path=getDrive()+"book\\gutenburg_BIN\\char_64",
            savepath=getDrive()+"Models\\PyTorch_v1\\Gutenburg\\",
            logpath=getDrive()+f'Model_Log\PyTorch\PTv{VERSION}_Gutenburg\\PTv{VERSION}_{datestr()}.txt',
            start=int( model_list[-1].split('__')[-1][:-3] )
            )
        
    else:
        #create model
        MODEL = PT_model_v1(meta_data=getDrive()+"book/gutenburg_BIN/metas/gutenburg_bin-RBT-char_meta_int64.pkl")
            
                    
        #------------------------
        #!! running
        prRed(Back.RED+f'TRAINING LEN: {len(os.listdir(getDrive()+"book/gutenburg_BIN/char_64"))}'+Style.RESET_ALL)
        input(Fore.CYAN+"Ready to run training? <ENTER>")
        
        MODEL.train_model(
            dir_path=getDrive()+"book\\gutenburg_BIN\\char_64",
            savepath=getDrive()+"Models\\PyTorch_v1\\Gutenburg\\",
            logpath=getDrive()+f'Model_Log\PyTorch\PTv{VERSION}_Gutenburg\\PTv{VERSION}_{datestr()}.txt'
            )