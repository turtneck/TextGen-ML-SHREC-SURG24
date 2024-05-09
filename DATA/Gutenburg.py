#/html/body/div[1]/div[1]/div[2]/div[4]/div/div[3]/div/table/tbody/tr[4]/td

import os,time,sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fun_colors import *

global dir_path
pr=False
dir_path = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__": print(f"DIRECTORY:\t\t{dir_path}>")

def folder_size(Folderpath):
    s=0
    for path, dirs, files in os.walk(Folderpath):
        for f in files:
            fp = os.path.join(path, f)
            s += os.path.getsize(fp)
    return s


#///////////////////////////////////////////////////////////////
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
driver = webdriver.Chrome()

waits=0.3
limit=10#gb
download_dir = "E:\ML_DATA" #dir_path
file_num=73556
curr_size= folder_size(download_dir+"/gutenburg")
 #get size of download dir
ar= os.listdir(download_dir+"/gutenburg")
if len(ar) ==0: start=1
else:
    #find max len
    br=ar.copy()
    l=0
    for k1 in ar:
        if len(k1)>l:l=len(k1)
    for k2 in ar:
        if len(k2)<l:br.remove(k2)
    start= int( br[-1][2:-4] )

#start=3513 #EMERGENCY OVERWRITE

for i in range(start,file_num):
    try:
        driver.get(f"https://www.gutenberg.org/ebooks/{i}")
        time.sleep(waits)
        #prCyan(f"PROG <{'{:.2f}'.format(float( i/file_num ))}%>")#, end =" ");pr=False
        sys.stdout.write(f"{Fore.CYAN}PROG {i}: <{'{:.2f}'.format(float( i/file_num ))}%>  ");pr=False
        for j in range(10):
            try:
                element = driver.find_element(By.XPATH, f"/html/body/div[1]/div[1]/div[2]/div[4]/div/div[3]/div/table/tbody/tr[{j}]/td")
                if element.text == "English":
                    #download
                    
                    driver.find_element(By.LINK_TEXT, 'Plain Text UTF-8').click()
                    time.sleep(waits)
                    #with open(dir_path+f"/gutenburg/pq{i}.txt", "w", encoding='utf-8') as f:
                    with open(download_dir+f"/gutenburg/pq{i}.txt", "w", encoding='utf-8') as f:
                        f.write(driver.page_source)
                    
                    #size
                    size = os.path.getsize(download_dir+f"/gutenburg/pq{i}.txt")
                    curr_size += size
                    prGreen(f"DL: {'{:.2f}'.format(float( size/(10**6) ))}MB: <{'{:.2f}'.format(float(  curr_size/(limit*(10**9))  ))}% full> \t {download_dir+f'/gutenburg/pq{i}.txt'}");pr=True
                    if curr_size > limit*(10**9):
                        print(Back.RED+f"SIZE LIMIT EXCEEDED"+Style.RESET_ALL)
                        os.exit()
                                           
                    break
            except:
                continue
        if not pr: prCyan("<!!>Not Accepted<!!>")
    except:
        continue



# driver.get("https://www.gutenberg.org/ebooks/2")
# time.sleep(1)
# for j in range(10):
#     try:
#         element = driver.find_element(By.XPATH, f"/html/body/div[1]/div[1]/div[2]/div[4]/div/div[3]/div/table/tbody/tr[{j}]/td")
#         if element.text == "English":
#             #download
#             driver.find_element(By.LINK_TEXT, 'Plain Text UTF-8').click()
#             time.sleep(1)
#             with open(dir_path+"/gutenburg/pq21059.txt", "w", encoding='utf-8') as f:
#                 f.write(driver.page_source)
#     except:
#         continue
