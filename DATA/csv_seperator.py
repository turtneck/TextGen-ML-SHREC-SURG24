import pandas as pd
import os
import re

global dir_path
dir_path = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__": print(f"DIRECTORY:\t\t{dir_path}>")

text_df = pd.read_csv(dir_path+"/fake_or_real_news.csv")
text = list(text_df.text)

cnt=0
for i in text:
    with open(dir_path+f'/init_news/news{cnt}.txt', 'w') as f:
        line = re.sub(r'[^A-Za-z0-9 ]+', '', i)
        line.replace('\n', '')
        f.write(line)
    cnt+=1

print("DONE!!!!!")



