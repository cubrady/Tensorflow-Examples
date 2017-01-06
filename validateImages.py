import os
import sys
import time
import argparse
import progressbar
import subprocess
from shutil import copyfile
from util import *

MODEL_NAME = 'output_graph.pb'
LABEL_NAME = 'output_labels.txt'

def __createFolders(workspace, label_lines, validate_folder):
    fset = validate_folder.split("/")
    optFolderBase = os.path.join(workspace, fset[-1] if fset[-1] else fset[-2])
    checkFolder(optFolderBase)
    for l in label_lines:
        optFolder = os.path.join(optFolderBase, l)
        checkFolder(optFolder)
    return optFolderBase

def __printResult(workspace, label_lines, dicResult, totalSpend, totalCount, printDetail = False):
    optFile = open(os.path.join(workspace, "result.txt"), "w")
    logAndWriteFile(optFile, "*" * 100)

    maxLabelLen = 0
    for l in label_lines:
        if len(l) > maxLabelLen:
            maxLabelLen = len(l)

    labelP = "%%%ds" % maxLabelLen
    for label, lstImage in dicResult.iteritems():
        logAndWriteFile(optFile, "Label " + labelP % label + " : %d ( %.2f%%)" % (len(lstImage), 100 * len(lstImage) / float(totalCount)))
    logAndWriteFile(optFile, "Total %d images" % (totalCount))

    logAndWriteFile(optFile, "*" * 100)
    logAndWriteFile(optFile, "Spend :%d sec, avg:%.3f sec, total process:%d" % (totalSpend, totalSpend/float(totalCount), totalCount))

    if printDetail:
        for label, lstImage in dicResult.iteritems():
            logAndWriteFile(optFile, " >>>>> Category %s" % label)
            for img, score in lstImage:
                logAndWriteFile(optFile, "%s:%f" % (img, score))

def __validateImages(workspace, validate_folder, limit = sys.maxint):
    # Creates graph from saved GraphDef.
    from retrainingExample import *
    create_graph(os.path.join(workspace, MODEL_NAME))
    label_lines = load_labels(os.path.join(workspace, LABEL_NAME))

    imageList = os.listdir(validate_folder)
    imageList = sorted(imageList, reverse=True)
    total = len(imageList)
    if limit < total:
        imageList = imageList[:limit]
        total = len(imageList)

    optFolder = __createFolders(workspace, label_lines, validate_folder)

    start = time.time()
    dicResult = {}
    for l in label_lines:
        dicResult[l] = []

    bar = progressbar.ProgressBar()
    for img in bar(imageList):
        imagePath = os.path.join(validate_folder, img)
        try:
            result = analyzeIamge(imagePath, label_lines)
        except:
            print "[Err] Invalid image path : %s" % imagePath
            continue

        #print "%s : %s" % (img , answer)

        answer, score = result[0]
        a2, s2 = result[1]
        dicResult[answer].append((img, score))

        copyfile(imagePath, os.path.join(optFolder, answer, "%d_%s%d_%s.jpg" % (score * 1000, a2, s2*1000, img.split(".")[0])))
        #break

    totalSpend = time.time() - start

    __printResult(workspace, label_lines, dicResult, totalSpend, total)

def __checkIfFolderValid(workspace, validate_folder):
    if not workspace or not os.path.exists(workspace):
        print "[Err]workspace:", args.workspace
        return False
    elif not validate_folder or not os.path.exists(validate_folder):
        print "[Err]validate_folder:", args.validate_folder
        return False
    return True

def batchValidateImages():
    lstWorkspace = [
        ('/home/brad_chang/deep_learning/trainedModel/tf/pg_hairstyle/v1/', "/data/dataset/training/pg1210_12199/"),
        ('/home/brad_chang/deep_learning/trainedModel/tf/fashinon_recog/v1/', "/data/dataset/training/pg1210_12199/"),
        ('/home/brad_chang/deep_learning/trainedModel/tf/pg_food/v1/', "/data/dataset/training/pg1210_12199/"),
        ('/home/brad_chang/deep_learning/trainedModel/tf/pg_text/v1/', "/data/dataset/training/pg1210_12199/"),
        ('/home/brad_chang/deep_learning/trainedModel/tf/xxx_recog_v2/v1/', "/data/dataset/training/pg1210_12199/"),
        ]
    for workspace, validate_folder in lstWorkspace:
        if not __checkIfFolderValid(workspace, validate_folder):
            return

    for workspace, validate_folder in lstWorkspace:
        launchValidateImageProcess(workspace, validate_folder)
        pass

def launchValidateImageProcess(workspace, validate_folder):
    print "Launch new python process ..."
    command = "python validateImages.py --new_process 1 --workspace %s --validate_folder %s" % (workspace, validate_folder)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print process.returncode

if __name__ == '__main__':
    #batchValidateImages()
    parser = argparse.ArgumentParser(sys.argv)
    parser.add_argument(
        "--batch",
        help="Batch Process"
    )
    parser.add_argument(
        "--workspace",
        help="The path of graph model and text"
    )
    parser.add_argument(
        "--validate_folder",
        help="Target validate folder"
    )
    parser.add_argument(
        "--new_process",
        help="Launch a new process to validate images"
    )

    args = parser.parse_args()
    if args.new_process:
        if __checkIfFolderValid(args.workspace, args.validate_folder):
            __validateImages(args.workspace, args.validate_folder)
    elif args.batch:
        batchValidateImages()
    else:
        if __checkIfFolderValid(args.workspace, args.validate_folder):
            __validateImages(args.workspace, args.validate_folder)
