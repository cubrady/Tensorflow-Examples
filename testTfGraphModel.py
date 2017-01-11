import os
import time
from shutil import copyfile
import argparse
import progressbar
import subprocess
from util import *
from config import *

##############################################################
# Path and test folder, analyzed image label predefinitions
##############################################################
TEST_FOLDER = "/home/brad_chang/deep_learning/dataset/test/label_set/"
MODEL_WORKSPACE = '/home/brad_chang/deep_learning/trainedModel/tf/pg_label_set/v3/'

DIC_LABEL_FOLDER_MAP = {
    # label : folder
    'fashion' : 'fashion',
    'hairstyle' : 'hairstyle',
    'food': "food",
    #'unknown' : "",
    'travel' : "travel",
    'text' : "text",
    'xxxnsfw80' : "xxx"}

##############################################################

OPT_FALSE_FOLDER = os.path.join(OPT_FOLDER, OPT_TEST_FOLDER, os.path.split(TEST_FOLDER)[-1])
MODEl_PATH = os.path.join(MODEL_WORKSPACE, TF_GRAPH_MODEL_NAME)
LABEL_PATH = os.path.join(MODEL_WORKSPACE, TF_GRAPH_LABEL_NAME)

OPT_RESULT_FOLDER = os.path.join(MODEL_WORKSPACE, "test_result_%s" % TEST_FOLDER.split("/")[-2])
OPT_FALSE_FOLDER = os.path.join(OPT_RESULT_FOLDER, "false_result")
OPT_FILE = os.path.join(OPT_RESULT_FOLDER, TEST_RESULT_FILE)

SCORE_CRITERIA = 0.5

def createResultOptFolder(folder, label):
    path = os.path.join(folder, label)
    checkFolder(path)
    return path

def testImages(label_lines, scoreCriteria = SCORE_CRITERIA):
    import progressbar
    from retrainingExample import create_graph, analyzeIamge
    create_graph(MODEl_PATH)
    checkFolder(OPT_FALSE_FOLDER)

    optLog = open(OPT_FILE, 'w')

    logAndWriteFile(optLog, "Model:%s" % MODEl_PATH)
    logAndWriteFile(optLog, "Test:%s" % TEST_FOLDER)

    folders = os.listdir(TEST_FOLDER)
    allStart = time.time()
    dicResult = {}
    totlImgCount = 0
    for folder in folders:
        folderPath = os.path.join(TEST_FOLDER, folder)
        if not os.path.isdir(folderPath):
            print "[Warning] %s is not a valid folder" % folderPath
            continue

        falseLabelFolder = createResultOptFolder(OPT_FALSE_FOLDER, folder)
        imageList = os.listdir(folderPath)

        logAndWriteFile(optLog, "\nStart to test %d images @ %s" % (len(imageList), os.path.join(TEST_FOLDER, folder)))
        logAndWriteFile(optLog, " >>>>>>>>>>>>>>>> Category %s" % folder)

        count = 0
        total = len(imageList)
        start = time.time()
        TP, FP, TN, FN = 0, 0, 0, 0
        bar = progressbar.ProgressBar()
        for img in bar(imageList):
            imgPath = os.path.join(folderPath, img)
            ans, score = analyzeIamge(imgPath, label_lines)[0] # Get top
            logAndWriteFile(optLog, "%s %s %.5f" % (imgPath, ans, score), printMsg = False)
            totlImgCount += 1
            if score > scoreCriteria:
                if ans == DIC_LABEL_FOLDER_MAP.get(folder):
                    TP += 1
                else:
                    TN += 1
            else:
                if ans == DIC_LABEL_FOLDER_MAP.get(folder):
                    FP += 1
                else:
                    FN += 1
                copyfile(imgPath, os.path.join(falseLabelFolder, "%.3f_%s_%s.jpg" % (score, ans, img.split(".jpg")[-2])))
            # if DIC_LABEL_FOLDER_MAP.get(ans) == folder:
            #     t += 1
            # else:
            #     f += 1
            #     copyfile(imgPath, os.path.join(falseLabelFolder, "%.3f_%s_%s.jpg" % (score, ans, img.split(".jpg")[-2])))

            count += 1

        dicResult[folder] = (TP, FP, TN, FN)

        # logAndWriteFile(optLog, "*" * 25 + " Result " + "*" * 25)
        # logAndWriteFile(optLog, "%s (count:%d) - T:%d(%.2f%%) F:%d(%.2f%%)" % (folder, total, t, t/float(total)*100, f, f/float(total)*100))
        # logAndWriteFile(optLog, "*" * 60)

    logAndWriteFile(optLog, "All process done, spend %s, %d images analyzed" % (formatSec(time.time() - allStart), totlImgCount))

    measureModelPerformance(optLog, dicResult)

def measureModelPerformance(optLog, dicResult, dicLabelSet = DIC_LABEL_FOLDER_MAP):
    logAndWriteFile(optLog, "\n" + "#" * 15 + " Model Performance Summary " + "#" * 15)

    def __calResult(TP, FP, TN, FN):
        pa, pb = TP, TP + FP
        precesion = 100.0 if pb == 0 else pa * 100 / float(pb)

        aa, ab = TP + TN, TP + FP + TN + FN
        accuracy = 100.0 if ab == 0 else (aa) * 100 / float(ab)

        ra, rb = TP, TP + FN
        recall = 100.0 if (rb) == 0 else ( ra * 100 / float(rb) )
        return ((precesion, pa, pb), (accuracy, aa, ab), (recall, ra, rb))

    def __printResult(TP, FP, TN, FN):
        precesionSet, accuracySet, recallSet = __calResult(TP, FP, TN, FN)
        logAndWriteFile(optLog, "Precision :\t%.2f%% (%d / %d)" % (precesionSet[0], precesionSet[1], precesionSet[2]) )
        logAndWriteFile(optLog, "Accuracy :\t%.2f%% (%d / %d)" % (accuracySet[0], accuracySet[1], accuracySet[2]))
        logAndWriteFile(optLog, "Recall :\t%.2f%% (%d / %d)" % ( recallSet[0], recallSet[1], recallSet[2] ))

    allTP, allFP, allTN, allFN = 0, 0, 0, 0
    for label, result in dicResult.iteritems():
        TP, FP, TN, FN = result
        logAndWriteFile(optLog, "-" * 21 + " %s " % label + "-" * 21)
        logAndWriteFile(optLog, "\tTotal\tTrue\tFalse")
        logAndWriteFile(optLog, "Pos\t%d\t %d\t %d" % (TP + FP, TP, FP))
        logAndWriteFile(optLog, "Neg\t%d\t %d\t %d" % (TN + FN, TN, FN))
        __printResult(TP, FP, TN, FN)

        allTP += TP
        allFP += FP
        allTN += TN
        allFN += FN

    logAndWriteFile(optLog, "=" * 21 + " Summary " + "=" * 21)
    __printResult(allTP, allFP, allTN, allFN)

def validateTfModel():
    from retrainingExample import load_labels
    label_lines = load_labels(LABEL_PATH)

    testFolders = os.listdir(TEST_FOLDER)
    valtestFolders = DIC_LABEL_FOLDER_MAP.values()
    canGoTest = True
    for folder in valtestFolders:
        if not folder or folder not in testFolders:
            print "[Err] Need to define '%s'" % folder
            canGoTest = False

    if canGoTest:
        testImages(label_lines)
        pass
    else:
        print "Labels : ", label_lines
        print "[Err] Lable Mapping Error !!!"
        print "[Err] Current defined label-folder map :\n%s" % DIC_LABEL_FOLDER_MAP

if __name__ == '__main__':
    validateTfModel()
    #test_nsfw_measureModelPerformance()
