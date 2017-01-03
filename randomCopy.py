import os
import random
from shutil import copyfile

def randomCopy(srcFolder, tarFolder, copyCount):
    files = os.listdir(srcFolder)
    fileCount = len(files)
    if fileCount <= copyCount:
        print fileCount, copyCount
        return files

    idxShown = []
    while len(idxShown) < copyCount:
        idx = random.randint(0, fileCount)
        if idx not in idxShown:
            idxShown.append(idx)

    print "Random complete, count:", len(idxShown)
    for idx in idxShown:
        path = os.path.join(srcFolder, files[idx])
        copyfile(path, os.path.join(tarFolder, files[idx]))
    print "Copy files complete"
