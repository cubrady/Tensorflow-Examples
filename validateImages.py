import os, time
from shutil import copyfile
from retrainingExample import *
from util import *

MODEL_NAME = 'output_graph.pb'
LABEL_NAME = 'output_labels.txt'
MODEL_VER = 3
WORKSPACE = '/home/brad_chang/deep_learning/trainedModel/tf/xxx_recog/v%d/' % MODEL_VER
MODEL_PATH = os.path.join(WORKSPACE, MODEL_NAME)
LABEL_PATH = os.path.join(WORKSPACE, LABEL_NAME)

#validate_folder = "/home/brad_chang/deep_learning/dataset/xxx_detection/train/xxx"
#validate_folder = "/home/brad_chang/deep_learning/dataset/xxx_detection/validate/"
validate_folder = "/home/brad_chang/deep_learning/dataset/photogrid_1208"

SAVE_TARGET_XXX = 1
SAVE_TARGET_NORAML = 2
CLASSIFY_RESULT_FOLDER_XXX = "mightBeXXX"
CLASSIFY_RESULT_FOLDER_NORMAL = "mightBeOK"

SAVE_TARGET = SAVE_TARGET_XXX
CLASSIFY_RESULT_FOLDER = CLASSIFY_RESULT_FOLDER_XXX if SAVE_TARGET == SAVE_TARGET_XXX else CLASSIFY_RESULT_FOLDER_NORMAL

def __checkSaveTarget(ans):
    if SAVE_TARGET == SAVE_TARGET_XXX:
        return "xxx" in ans
    else:
        return "xxx" not in ans

def validateImages():
    # Creates graph from saved GraphDef.
    create_graph(MODEL_PATH)

    imageList = os.listdir(validate_folder)
    imageList = sorted(imageList, reverse=True)

    optFolderBase = os.path.join(WORKSPACE, CLASSIFY_RESULT_FOLDER)
    checkFolder(optFolderBase)
    optFolder = os.path.join(optFolderBase, os.path.split(validate_folder)[1])
    checkFolder(optFolder)

    f = open(LABEL_PATH, 'rb')
    label_lines = f.readlines()

    count = 0
    total = len(imageList)
    start = time.time()
    lstXXX = []
    lstNormal = []
    for img in imageList:
        imagePath = os.path.join(validate_folder, img)
        try:
            answer, score = analyzeIamge(imagePath, label_lines)
        except:
            print "[Err] Invalid image path : %s" % imagePath
            continue

        print "%s : %s" % (img , answer)

        if "xxx" in answer:
            lstXXX.append((img, score))
        else:
            lstNormal.append((img, score))

        if __checkSaveTarget(answer):
            copyfile(imagePath, os.path.join(optFolder, "%.3f_%s.jpg" % (score, img.split(".")[0])))

        count += 1

        printProgress(start, count, total)

    totalSpend = time.time() - start

    print "*" * 100
    def __dump(cat, lst):
        print " >>>>> Category %s" % cat
        lst = sorted(lst, key = lambda x:x[1])
        for img, score in lst:
            print img, score
    __dump("xxx", lstXXX)
    __dump("normal", lstNormal)

    print "[Result] Total : %d, xxx:%d, normal:%d" % (total, len(lstXXX), len(lstNormal))


    print "*" * 100
    print "Spend :%d sec, avg:%.3f sec, total process:%d" % (totalSpend, totalSpend/float(total), total)

if __name__ == '__main__':
    validateImages()
