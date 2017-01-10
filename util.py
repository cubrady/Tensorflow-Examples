import os
import time

def loadLables(path):
    f = open(path, 'rb')
    label_lines = f.readlines()
    f.close()
    return label_lines

def checkFolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def printProgress(start, count, total, whenToPrint = 10):
    if count % whenToPrint == 0:
        print "=" * 100
        spend = time.time() - start
        spendTime = spend / float(count)
        remain = (total - count) * spendTime
        print "[%.2f%%] %d/%d, remain %s, avg:%.3f" % (count / float(total) * 100, count, total, formatSec(remain), spendTime)
        print "=" * 100
        return True
    return False

def formatSec(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def logAndWriteFile(f, msg):
    if f : f.write(msg + "\n")
    print msg

PIL_IMG_FORMAT_JPEG = "JPEG"
PIL_IMG_FORMAT_PNG = "PNG"
def ensureImageFormatValid(path, validImageSet = [PIL_IMG_FORMAT_JPEG, PIL_IMG_FORMAT_PNG], delInvaidFile = False):
    # /data/dataset/training/flower_photos/
    from PIL import Image
    for dirPath, dirNames, fileNames in os.walk(path):
        print "Scanning %s ... " % dirPath
        for f in fileNames:
            img_path = os.path.join(dirPath, f)
            try:
                im = Image.open(img_path)
                if im.format not in validImageSet:
                    print img_path
            except Exception as err:
                print err
                print "ERROR : %s" % img_path
                if delInvaidFile:
                    os.remove(img_path)
                    print "Remove %s OK" % img_path
