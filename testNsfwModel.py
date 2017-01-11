def test_measureModelPerformance():
    optLog = None#open("", 'w')
    dicLabelSet = DIC_LABEL_FOLDER_MAP
    dicResult = {"xxx":(13, 20), "normal":(123, 1)}
    measureModelPerformance(optLog, dicResult, dicLabelSet)

def test_nsfw_measureModelPerformance():
    optLog = None#open("", 'w')
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
