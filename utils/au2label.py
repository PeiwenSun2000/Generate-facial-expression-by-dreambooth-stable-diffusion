import os
import glob
import pandas as pd
import numpy as np
import tqdm as tqdm
file_path="/home/DataBase1/sunpeiwen/dataset/BP4D-512-crop/pics"
types = ('*.jpg', '*.png', '*.jpeg', '*.gif', '*.webp', '*.bmp') 
files_grabbed = []
for files in types:
    files_grabbed.extend(glob.glob(os.path.join(file_path, files)))
files_grabbed = sorted(files_grabbed)

with open('/home/DataBase1/zwd/BP4D-crop/all_imgs.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

for idx,ann in enumerate(data):
    data[idx]=ann.strip('\n')
data=sorted(data)
new_data={}
for ann in data:
    temp=ann.split(":")
    new_data[temp[0]]=temp[1].split(',')

data = pd.DataFrame(new_data).transpose()

for i in range(12):
    data[i] = np.where(data[i] == "1","AU_"+str(i),None)
 
assert len(files_grabbed)==len(new_data)

i = 1
for index, row in data.iterrows():
    line_con = ",".join(list(filter(None,list(row))))
    txt_filename = os.path.join("/home/DataBase1/sunpeiwen/dataset/BP4D-512-crop/au-label", f"{str(i).zfill(6)}.txt")
    i+=1
    with open(txt_filename, 'w') as f:
        f.write(line_con)
