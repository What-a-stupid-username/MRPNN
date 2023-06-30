from PIL import Image
import glob
import os

def Search(rootDir, suffixes):
    res = []
    for lists in os.listdir(rootDir):       
        path = os.path.join(rootDir, lists)
        if os.path.isfile(path):
            if path.endswith(suffixes):
                res.append(path)
        if os.path.isdir(path):
            res += Search(path, suffixes)
    return res

files = Search(path, ('.bmp', '.BMP'))

for img in files:
    Image.open(img).save(img.replace('.bmp','.png'))