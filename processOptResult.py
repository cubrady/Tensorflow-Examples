import os
import operator
import collections
import time

from datetime import datetime
from config import *

def __classifyImagesByFolder(root, folder, get_inference_on_image):
    t = time.time()
    dicDetailResult = {}
    dicResult = {}
    imageList = os.listdir(folder)
    cnt = 0
    length = len(imageList)
    for img in imageList:
        imgPath = os.path.join(folder, img)
        if os.path.isdir(imgPath):
            __classifyImagesByFolder(root, imgPath, get_inference_on_image)
            continue

        dicInference = get_inference_on_image(imgPath)
        dicDetailResult[img] = dicInference
        for label, prob in dicInference.iteritems():
            if label in dicResult:
                acumProb, cnt = dicResult[label]
                dicResult[label] = (acumProb + prob, cnt + 1)
            else:
                dicResult[label] = (prob, 1)

        cnt += 1
        fd = folder.split(root)
        if len(fd) > 1 and cnt == length:#and len(fd[1].split("/")) > 2:
            progress_msg = "Progress Overview (%s): %.2f%% (%d/%d)" % (fd[1], cnt/float(length)*100, cnt, length)
            # http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
            print('\x1b[3;33;40m' + progress_msg + '\x1b[0m')
        if cnt % 10 == 0 or cnt == length:
            progress_msg = "Progress (%s): %.2f%% (%d/%d)" % (fd[1], cnt/float(length)*100, cnt, length)
            print('\x1b[0;33;40m' + progress_msg + '\x1b[0m')


    if len(dicResult) > 0:

        spend = (time.time() - t)
        msg = ">>>>> %d images scan complete, %d labels found, spend %.3f sec, avg:%.1fms" % (len(imageList), len(dicResult), spend, 1000 * spend / float(len(imageList)))
        print (msg)

        lstResult = sorted(dicResult.items(), key=operator.itemgetter(1))
        lstResult.reverse()
        #print (lstResult)

        #print root, folder
        #print folder.split(root)
        save_folder = os.path.split(root)[-1]
        filename = folder.split(root)[1][1:]
        #print ("save_folder:", save_folder, ", filename:", filename)
        __saveFile(filename, save_folder, "", lstResult, msg)
        __saveFile(filename, save_folder, "_detail", dicDetailResult, msg)

def classifyImagesByFolder(folder, get_inference_on_image):
    __classifyImagesByFolder(folder, folder, get_inference_on_image)

def __saveFile(filename, save_folder, postfix, data, msg):
    abs_save_folder = os.path.join(os.path.dirname(__file__), save_folder)
    save_file = os.path.join(abs_save_folder, filename) + postfix + ".txt"
    print "save file path:", save_file
    tar_folder = os.path.split(save_file)[0]
    if not os.path.exists(tar_folder):
        os.makedirs(tar_folder)
    f = open(save_file, 'w')
    f.write("%s\n" % str(datetime.now()))
    f.write("folder:%s\n" % save_folder)
    f.write(msg + "\n\n")

    if isinstance(data, list):
        for k, v in data:
            #print (k, v)
            # ret = ""
            # for item in v:
            #     ret += "%s\t" % str(item)
            #print "%s\t%s\n" % (k, ret[:len(ret)-1])
            f.write("%s\t%s\n" % (k, v))
    elif isinstance(data, dict):
        for k, v in data.iteritems():
           f.write("%s\t%s\n" % (k, v))
    f.close()

def parseString(line):
    #print line.split("\t")
    filename, dicResult = line.split("\t")
    dicResult = eval(dicResult)
    #print filename, dicResult
    return filename, dicResult

def parseDetailFileToAvgLable(filePath, probCriteria = 0.0):
    # Open detail file
    f = open(filePath, "r")
    #print filePath
    allLines = f.readlines()
    f.close()
    if len(allLines) <= 4:
        # this might be a invalid file
        return

    dicLabelAvgProb = {}
    allLines = allLines[4:]
    for line in allLines:
        filename, dicInference = parseString(line)
        for label, prob in dicInference.iteritems():
            if prob <= probCriteria:
                continue

            if label in dicLabelAvgProb:
                acumProb, cnt = dicLabelAvgProb[label]
                dicLabelAvgProb[label] = (acumProb + prob, cnt + 1)
            else:
                dicLabelAvgProb[label] = (prob, 1)

    for k, v in dicLabelAvgProb.iteritems():
        prob, cnt = v
        if cnt > 1:
            dicLabelAvgProb[k] = (prob/float(cnt), cnt)
    #print dicLabelAvgProb.items()
    lstResult = sorted(dicLabelAvgProb.items(), key=operator.itemgetter(1))
    lstResult.reverse()
    #print lstResult
    lstResult = sorted(lstResult, key=lambda tup: tup[1][1])
    lstResult.reverse()
    #print lstResult

    optpath, name = os.path.split(filePath.replace("_detail.txt", ""))
    __saveFile(name + "_labelAvg_PC%.1f" % probCriteria, optpath, "", lstResult, optpath)

    return dicLabelAvgProb

def getAllDetailFiles(folder):
    lstFiles = []
    flist = os.listdir(folder)
    for f in flist:
        if "_detail" in f and "labelAvg" not in f:
            lstFiles.append(os.path.join(folder, f))
    return lstFiles

def checkLabelOverlap(dicLabel1, dicLabel2):
    a_multiset = collections.Counter(dicLabel1.keys())
    b_multiset = collections.Counter(dicLabel2.keys())
    lstOverlap = list((a_multiset & b_multiset).elements())
    #print lstOverlap
    print "Overlap count:%d" % len(lstOverlap)
    dicOverlap = {}
    for ol in lstOverlap:
        dicOverlap[ol] = (dicLabel1[ol], dicLabel2[ol])
    return dicOverlap

def checkLabelOverlapWithProb(prob, lstAllLabels, writeFile):
    def __printAndWriteFile(msg):
        print msg
        writeFile.write(msg + "\n")

    __printAndWriteFile(">>>>>>>>> Check label overlapping that with prob > %.2f" % prob)

    lstAllLabelsFilterProb = []
    for dicLabels in lstAllLabels:
        dicLabelsProb = {}
        for k, v in dicLabels.iteritems():
            #print v
            if v[0] >= prob:
                dicLabelsProb[k] = v

        __printAndWriteFile("%s label count : %d / %d" % ("", len(dicLabelsProb), len(dicLabels)))

        lstAllLabelsFilterProb.append(dicLabelsProb)

    dicOverlap = {}
    for i in xrange(0, len(lstAllLabelsFilterProb) - 1):
        dicOverlap = checkLabelOverlap(lstAllLabelsFilterProb[i], lstAllLabelsFilterProb[i + 1])

    __printAndWriteFile("Overlap count:%d" % len(dicOverlap))

    for label, prob in dicOverlap.iteritems():
        writeFile.write(label + "\t" + str(prob) + "\n")

def calculateLabelOverlap(lstAllLabels, folder, probCriteria):
    save_file = os.path.join(os.path.dirname(__file__), folder, "overlap_%.1f.txt" % probCriteria)
    print "overlap file:", save_file
    writeFile = open(save_file, "w")
    for idx in xrange(9, -1, -1):
        checkLabelOverlapWithProb(idx/float(10), lstAllLabels, writeFile)
        writeFile.write("\n")
    writeFile.close()

def __calLabelAvg(folder, probCriteria, mergeResult = True):
    lstFiles = getAllDetailFiles(folder)
    lstAllLabels = []
    for f in lstFiles:
        dicLabelAvgProb = parseDetailFileToAvgLable(f, probCriteria)
        if dicLabelAvgProb:
            #print dicLabelAvgProb
            #print "-" * 100
            lstAllLabels.append(dicLabelAvgProb)
            print "%s label count : %d" % (f, len(dicLabelAvgProb))

    #calculateLabelOverlap(lstAllLabels, folder, 0.0)

    if mergeResult:
        dicAllLabelAvg = {}
        for dicLabelAvg in lstAllLabels:
            for label, probSet in dicLabelAvg.iteritems():
                if dicAllLabelAvg.has_key(label):
                    prob, cnt = dicAllLabelAvg[label]
                    dicAllLabelAvg[label] = (prob + probSet[0], cnt + probSet[1])
                else:
                    dicAllLabelAvg[label] = probSet

        lstResult = sorted(dicLabelAvgProb.items(), key=operator.itemgetter(1))
        lstResult = sorted(lstResult, key=lambda tup: tup[1][1])
        lstResult.reverse()

        name = os.path.split(folder)[-1]
        __saveFile(name + "_AllLabelAvg_PC%.1f" % probCriteria, folder, "", lstResult, folder)

def __getAllFolders(root):
    flist = os.listdir(root)
    lsdFolders = []
    for f in flist:
        p = os.path.join(root, f)
        if os.path.isdir(p):
            lsdFolders.append(p)
    return lsdFolders

def calLabelAvg(folder):
    probCriteria = 0.3
    # lsdFolders = __getAllFolders(folder)
    # print lsdFolders
    # for f in lsdFolders:
    #     __calLabelAvg(f)
    #     break
    __calLabelAvg(os.path.join(folder, "fashion"), probCriteria)


if __name__ == '__main__':
    calLabelAvg(OPT_FOLDER)
