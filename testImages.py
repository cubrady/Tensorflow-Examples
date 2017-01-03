import os
import time
from shutil import copyfile
from retrainingExample import *
from util import *
from config import *

TEST_FOLDER = "/home/brad_chang/deep_learning/dataset/test/fashion_test_1.5k"
OPT_FALSE_FOLDER = os.path.join(OPT_FOLDER, OPT_TEST_FOLDER, os.path.split(TEST_FOLDER)[-1])
OPT_FILE = OPT_FALSE_FOLDER + "_test_opt.txt"

VERSION = 1
WORKSPACE = '/home/brad_chang/deep_learning/trainedModel/tf/fashinon_recog/v%d' % VERSION
MODEl_PATH = os.path.join(WORKSPACE, "output_graph.pb")
LABEL_PATH = os.path.join(WORKSPACE, "output_labels.txt")

DIC_LABEL = {"pos":"fashion", "neg":"nofashion"}

def logAndWriteFile(f, msg):
    if f : f.write(msg + "\n")
    print msg

def createFalseLabelFolder(label):
    path = os.path.join(OPT_FALSE_FOLDER, label)
    checkFolder(path)
    return path

def testImages():
    create_graph(MODEl_PATH)
    checkFolder(OPT_FALSE_FOLDER)

    label_lines = loadLables(LABEL_PATH)

    optLog = open(OPT_FILE, 'w')

    logAndWriteFile(optLog, "Model:%s\n" % MODEl_PATH)

    folders = os.listdir(TEST_FOLDER)
    allStart = time.time()
    dicResult = {}
    for folder in folders:
        if folder.startswith("."):
            continue
        falseLabelFolder = createFalseLabelFolder(folder)
        folderPath = os.path.join(TEST_FOLDER, folder)
        imageList = os.listdir(folderPath)

        logAndWriteFile(optLog, "\nStart to test %d images @ %s" % (len(imageList), os.path.join(TEST_FOLDER, folder)))
        logAndWriteFile(optLog, " >>>>>>>>>>>>>>>> Category %s \n" % folder)

        count = 0
        total = len(imageList)
        start = time.time()
        t, f = 0, 0
        for img in imageList:
            if img.startswith("."):
                continue
            imgPath = os.path.join(folderPath, img)
            ans, score = analyzeIamge(imgPath, label_lines)
            logAndWriteFile(optLog, "%s %s %.5f" % (img, ans, score))
            if ans == folder:
                t += 1
            else:
                f += 1
                copyfile(imgPath, os.path.join(falseLabelFolder, "%.3f_%s_%s.jpg" % (score, ans, img.split(".")[0])))

            count += 1
            if printProgress(start, count, total):
                print "%s (count:%d) - T:%d(%.2f%%) F:%d(%.2f%%)" % (folder, total, t, t/float(total)*100, f, f/float(total)*100)

        dicResult[folder] = (t, f)

        logAndWriteFile(optLog, "*" * 25 + " Result " + "*" * 25)
        logAndWriteFile(optLog, "%s (count:%d) - T:%d(%.2f%%) F:%d(%.2f%%)" % (folder, total, t, t/float(total)*100, f, f/float(total)*100))
        logAndWriteFile(optLog, "*" * 60)

    logAndWriteFile(optLog, "All process done, spend %s" % formatSec(time.time() - allStart))

    measureModelPerformance(optLog, dicResult)

def measureModelPerformance(optLog, dicResult, dicLabelSet = DIC_LABEL):
    logAndWriteFile(optLog, "\n" + "#" * 15 + " Model Performance Summary " + "#" * 15)

    positiveLabel = dicLabelSet.get("pos")
    negativeLabel = dicLabelSet.get("neg")
    TP, FP = dicResult.get(positiveLabel)
    TN, FN = dicResult.get(negativeLabel)

    logAndWriteFile(optLog, "-" * 42)
    logAndWriteFile(optLog, "\t\tTotal\tTrue\tFalse")
    logAndWriteFile(optLog, "%12s\t %d\t %d\t %d" % (positiveLabel, TP + FP, TP, FP))
    logAndWriteFile(optLog, "%12s\t %d\t %d\t %d" % (negativeLabel, TN + FN, TN, FN))
    logAndWriteFile(optLog, "-" * 42)

    logAndWriteFile(optLog, "Precision :\t%.2f%%" % ( TP * 100 / float(TP + FP)) )
    logAndWriteFile(optLog, "Accuracy :\t%.2f%%" % ( (TP + TN) * 100 / float(TP + FP + TN + FN) ))
    logAndWriteFile(optLog, "Recall :\t%.2f%%" % ( TP * 100 / float(TP + FN) ))

if __name__ == '__main__':
    testImages()
    #optLog = None
    #dicResult = {"fashion":(72, 28), "nofashion":(1142, 258)}
    #measureModelPerformance(None, dicResult)
