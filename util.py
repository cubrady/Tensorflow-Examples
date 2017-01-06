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
