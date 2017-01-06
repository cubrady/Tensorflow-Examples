import os
import time
from shutil import copyfile
from retrainingExample import *
from util import *
from config import *

# Modify these two pathes !!
TEST_FOLDER = "/home/brad_chang/deep_learning/dataset/test/xxx"
MODEL_WORKSPACE = '/home/brad_chang/deep_learning/trainedModel/tf/xxx_recog_v2/v1'

OPT_FALSE_FOLDER = os.path.join(OPT_FOLDER, OPT_TEST_FOLDER, os.path.split(TEST_FOLDER)[-1])
MODEl_PATH = os.path.join(MODEL_WORKSPACE, "output_graph.pb")
LABEL_PATH = os.path.join(MODEL_WORKSPACE, "output_labels.txt")

label_lines = load_labels(LABEL_PATH)

OPT_RESULT_FOLDER = os.path.join(MODEL_WORKSPACE, "result")
OPT_FALSE_FOLDER = os.path.join(OPT_RESULT_FOLDER, "false_result")
OPT_FILE = os.path.join(OPT_RESULT_FOLDER, "test_result.txt")

DIC_LABEL = { \
    "pos" : "xxx" ,\
    "neg" : "normal" }

DIC_LABEL_FOLDER_MAP = { \
    "xxx nsfw 0 8" : DIC_LABEL.get("pos"), \
    "pg nsfw 0 02" : DIC_LABEL.get("neg") }

def createFalseLabelFolder(label):
    path = os.path.join(OPT_FALSE_FOLDER, label)
    checkFolder(path)
    return path

def testImages():
    create_graph(MODEl_PATH)
    checkFolder(OPT_FALSE_FOLDER)

    optLog = open(OPT_FILE, 'w')

    logAndWriteFile(optLog, "Model:%s" % MODEl_PATH)
    logAndWriteFile(optLog, "Test:%s" % TEST_FOLDER)

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
            ans, score = analyzeIamge(imgPath, label_lines)[0]
            logAndWriteFile(optLog, "%s %s %.5f" % (img, ans, score))
            if DIC_LABEL_FOLDER_MAP.get(ans) == folder:
                t += 1
            else:
                f += 1
                copyfile(imgPath, os.path.join(falseLabelFolder, "%.3f_%s_%s.jpg" % (score, ans, img.split(".jpg")[-2])))

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

def test_measureModelPerformance():
    optLog = None#open("", 'w')
    dicLabelSet = DIC_LABEL
    dicResult = {"xxx":(13, 20), "normal":(123, 1)}
    measureModelPerformance(optLog, dicResult, dicLabelSet)

def test_nsfw_measureModelPerformance():
    optLog = None#open("", 'w')
    #dicLabelSet = DIC_LABEL
    posFolder = "/home/brad_chang/deep_learning/dataset/test/xxx/xxx"
    negFolder = "/home/brad_chang/deep_learning/dataset/test/xxx/normal"

    def __getTFResult(path, rev = False, creteria = 0.9):
        lst = os.listdir(path)
        t, f = 0, 0
        for img in lst:
            #print img.split("_")
            score, name = img.split("_")
            if float(score) >= creteria:
                t += 1
            else:
                f += 1
        return (f, t) if rev else (t, f)

    for cri in [0.7, 0.75, 0.8, 0.85, 0.95]:
        print "Criteria : ", cri
        dicResult = {"xxx" : __getTFResult(posFolder, creteria = cri), "normal" : __getTFResult(negFolder, rev = True, creteria = cri)}
        measureModelPerformance(optLog, dicResult)
        print "\n\n"

def nsfwFilter():
    pass

if __name__ == '__main__':
    if not DIC_LABEL_FOLDER_MAP.has_key(label_lines[0]):
        print "Lable Mapping Error !!!"
        print "Need to define ", label_lines
        print "Current defined labels:", DIC_LABEL_FOLDER_MAP
    else:
        testImages()

    #test_nsfw_measureModelPerformance()
