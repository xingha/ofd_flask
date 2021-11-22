import os

paths = []
root = r'D:\data\OCR-labeled\50007-2'
name = r'\Label.txt'
filn = root + name
with open(filn,encoding="utf-8") as f:
    lines = f.readlines()
    f.writelines()
for line in lines:
    filepath = line.split('\t')[0].split('/')[-1]
    if not os.path.exists(root+'/'+filepath):
        print('--->',root+'/'+filepath)
        os.remove(root+'/'+filepath)
    else:
        print('.....ok')
