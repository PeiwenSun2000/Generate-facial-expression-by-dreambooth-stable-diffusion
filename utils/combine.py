import os
import glob
from tqdm import tqdm
file_path="/home/DataBase1/sunpeiwen/dataset/BP4D-512-crop/au-label"
file_path2="/home/DataBase1/sunpeiwen/dataset/BP4D-512-crop/ddbr-label"
file_path3="/home/DataBase1/sunpeiwen/dataset/BP4D-512-crop/combine-label"
types = ('*.txt')
files_grabbed = []
for files in types:
    files_grabbed.extend(glob.glob(os.path.join(file_path, files)))
files_grabbed = sorted(files_grabbed)
files_grabbed = files_grabbed[1:]
for files_ in tqdm(files_grabbed):
    with open(files_) as f:
        line = f.readline()
        line=line.strip("\n")
        f.close()
    with open(os.path.join(file_path2,os.path.basename(files_))) as f:
        line2 = f.readline()
        line2=line2.strip("\n")
        f.close()
    with open(os.path.join(file_path3,os.path.basename(files_)),'w') as f:
        f.write(line+","+line2)
