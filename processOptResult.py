import os
import operator
import collections
import time

from datetime import datetime
from config import *

def classifyImagesByFolder(folder, get_inference_on_image):
    t = time.time()
    dicDetailResult = {}
    dicResult = {}
    imageList = os.listdir(folder)
    cnt = 0
    for img in imageList:
        imgPath = os.path.join(folder, img)
        if os.path.isdir(imgPath):
            classifyImagesByFolder(imgPath, get_inference_on_image)
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
        # if cnt > 5:
        #     break

    msg = "%d images scan complete, %d labels found, spend %.3f sec" % (len(imageList), len(dicResult), (time.time() - t))
    print (msg)

    lstResult = sorted(dicResult.items(), key=operator.itemgetter(1))
    lstResult.reverse()
    print (lstResult)

    __saveFile(folder, "", lstResult, msg)
    __saveFile(folder, "_detail", dicDetailResult, msg)

def __saveFile(folder, postfix, data, msg):
    save_file = os.path.join(os.path.dirname(__file__), OPT_FOLDER, os.path.split(folder)[-1]) + postfix + ".txt"
    print ("save_file:", save_file)
    f = open(save_file, 'w')
    f.write("%s\n" % str(datetime.now()))
    f.write("folder:%s\n" % folder)
    f.write(msg + "\n\n")

    if isinstance(data, list):
        for k, v in data:
            #print (k, v)
            ret = ""
            for item in v:
                ret += "%s\t" % str(item)
            #print "%s\t%s\n" % (k, ret[:len(ret)-1])
            f.write("%s\t%s\n" % (k, ret[:len(ret)-1]))
    elif isinstance(data, dict):
        for k, v in data.iteritems():
           #print (k, v)
           ret = ""
           for item in v:
               ret += "%s\t" % str(item)
           #print ret
           f.write("%s\t%s\n" % (k, ret[:len(ret)-1]))
    f.close()

def parseString(line):
    #print line
    filename, dicResult = line.split("\t")
    dicResult = eval(dicResult)
    #print filename, dicResult
    return filename, dicResult

def parseDetailFileToAvgLable(filePath, probCriteria = 0.0):
    # Open detail file
    f = open(filePath, "r")
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
    lstResult = sorted(dicLabelAvgProb.items(), key=operator.itemgetter(1))
    lstResult.reverse()

    optpath = filePath.replace("_detail.txt", "")
    __saveFile(optpath, "_labelAvg_PC%.1f" % probCriteria, lstResult, optpath)

    return dicLabelAvgProb

def getAllDetailFiles():
    lstFiles = []
    flist = os.listdir(OPT_FOLDER)
    for f in flist:
        if "_detail" in f and "labelAvg" not in f:
            lstFiles.append(os.path.join(OPT_FOLDER, f))
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
    dicOverlap = checkLabelOverlap(lstAllLabelsFilterProb[0], lstAllLabelsFilterProb[1])
    __printAndWriteFile("Overlap count:%d" % len(dicOverlap))

    for label, prob in dicOverlap.iteritems():
        writeFile.write(label + "\t" + str(prob) + "\n")

if __name__ == '__main__':
    lstFiles = getAllDetailFiles()
    lstAllLabels = []
    probCriteria = 0.3
    for f in lstFiles:
        dicLabelAvgProb = parseDetailFileToAvgLable(f, probCriteria)
        if dicLabelAvgProb:
            lstAllLabels.append(dicLabelAvgProb)
            print "%s label count : %d" % (f, len(dicLabelAvgProb))

    if len(lstAllLabels) == 2:
        checkLabelOverlap(lstAllLabels[0], lstAllLabels[1])
        save_file = os.path.join(os.path.dirname(__file__), OPT_FOLDER, "overlap_%.1f.txt" % probCriteria)
        print "overlap file:", save_file
        writeFile = open(save_file, "w")
        for idx in xrange(9, -1, -1):
            checkLabelOverlapWithProb(idx/float(10), lstAllLabels, writeFile)
            writeFile.write("\n")
        writeFile.close()
