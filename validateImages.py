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
    optFolder = fset[-1] if fset[-1] else fset[-2]
    optFolderBase = os.path.join(workspace, optFolder)
    checkFolder(optFolderBase)
    for l in label_lines:
        folder = os.path.join(optFolderBase, l)
        checkFolder(folder)
    return optFolderBase, optFolder

def __printResult(workspace, label_lines, dicResult, totalSpend, totalCount, optFolderName, printDetail = False):
    optFilePath = os.path.join(workspace, "result_%s.txt" % optFolderName)
    optFile = open(optFilePath, "w")
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
    print "Result is dumpped to %s" % optFilePath

    if printDetail:
        for label, lstImage in dicResult.iteritems():
            logAndWriteFile(optFile, " >>>>> Category %s" % label)
            for img, score in lstImage:
                logAndWriteFile(optFile, "%s:%f" % (img, score))

def __validateImage(workspace, imagePath, testCount = 1):
    from retrainingExample import create_graph, load_labels, analyzeIamge
    create_graph(os.path.join(workspace, MODEL_NAME))
    label_lines = load_labels(os.path.join(workspace, LABEL_NAME))
    t = time.time()
    for i in xrange(0, testCount):
        result = analyzeIamge(imagePath, label_lines)
        print result
    print "analyzeIamge spend:%f sec" % ((time.time() - t) / float(testCount))

def __validateImages(workspace, validate_folder, copy = 1, limit = sys.maxint):
    # Creates graph from saved GraphDef.
    from retrainingExample import create_graph, load_labels, analyzeIamge
    create_graph(os.path.join(workspace, MODEL_NAME))
    label_lines = load_labels(os.path.join(workspace, LABEL_NAME))

    imageList = os.listdir(validate_folder)
    imageList = sorted(imageList, reverse=True)
    total = len(imageList)
    if limit < total:
        imageList = imageList[:limit]
        total = len(imageList)

    optFolder, optFolderName = __createFolders(workspace, label_lines, validate_folder)

    dicResult = {}
    for l in label_lines:
        dicResult[l] = []

    bar = progressbar.ProgressBar()

    start = time.time()
    for img in bar(imageList):
        imagePath = os.path.join(validate_folder, img)
        result = analyzeIamge(imagePath, label_lines)
        #print img, result
        if result:
            answer, score = result[0]
            a2, s2 = result[1]
            dicResult[answer].append((img, score))

        if copy: copyfile(imagePath, os.path.join(optFolder, answer, "%d_%s%d_%s.jpg" % (score * 1000, a2, s2*1000, img.split(".")[0])))
        #break

    totalSpend = time.time() - start

    __printResult(workspace, label_lines, dicResult, totalSpend, total, optFolderName)

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
        # ('/home/brad_chang/deep_learning/trainedModel/tf/pg_hairstyle/v1/', "/data/dataset/training/pg1210_12199/"),
        # ('/home/brad_chang/deep_learning/trainedModel/tf/fashinon_recog/v1/', "/data/dataset/training/pg1210_12199/"),
        # ('/home/brad_chang/deep_learning/trainedModel/tf/pg_food/v1/', "/data/dataset/training/pg1210_12199/"),
        # ('/home/brad_chang/deep_learning/trainedModel/tf/pg_text/v1/', "/data/dataset/training/pg1210_12199/"),
        ('/home/brad_chang/deep_learning/trainedModel/tf/xxx_recog_v2/v2/', "/data/dataset/training/pg1210_12199/"),
        ]
    for workspace, validate_folder in lstWorkspace:
        if not __checkIfFolderValid(workspace, validate_folder):
            return

    for workspace, validate_folder in lstWorkspace:
        launchValidateImageProcess(workspace, validate_folder)
        pass

def launchValidateImageProcess(workspace, validate_folder):
    print "Launch new python process ..."
    print "workspace:", workspace
    print "validate_folder:", validate_folder
    command = "python validateImages.py --new_process 1 --workspace %s --validate_folder %s" % (workspace, validate_folder)
    #process = subprocess.Popen(command)#, shell=True, stdout=subprocess.PIPE)
    subprocess.call(command, shell=True)
    #process.wait()

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
        "--validate_file",
        help="Target validate image file"
    )
    parser.add_argument(
        "--validate_folder",
        help="Target validate folder"
    )
    parser.add_argument(
        "--new_process",
        help="Launch a new process to validate images"
    )
    parser.add_argument(
        "--copyfile",
        help="Copy analyzed image to the folder with label name"
    )

    args = parser.parse_args()
    if args.validate_file:
        if __checkIfFolderValid(args.workspace, args.workspace):
            __validateImage(args.workspace, args.validate_file)
    elif args.new_process:
        if __checkIfFolderValid(args.workspace, args.validate_folder):
            __validateImages(args.workspace, args.validate_folder, copy = args.copyfile)
    elif args.batch:
        batchValidateImages()
    else:
        if __checkIfFolderValid(args.workspace, args.validate_folder):
            __validateImages(args.workspace, args.validate_folder, copy = args.copyfile)
