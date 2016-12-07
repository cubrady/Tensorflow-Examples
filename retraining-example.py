"""
Preparing model:
 - Install bazel ( check tensorflow's github for more info )
    Ubuntu 14.04:
        - Requirements:
            sudo add-apt-repository ppa:webupd8team/java
            sudo apt-get update
            sudo apt-get install oracle-java8-installer
        - Download bazel, ( https://github.com/bazelbuild/bazel/releases )
          tested on: https://github.com/bazelbuild/bazel/releases/download/0.2.0/bazel-0.2.0-jdk7-installer-linux-x86_64.sh
        - chmod +x PATH_TO_INSTALL.SH
        - ./PATH_TO_INSTALL.SH --user
        - Place bazel onto path ( exact path to store shown in the output)
- For retraining, prepare folder structure as
    - root_folder_name
        - class 1
            - file1
            - file2
        - class 2
            - file1
            - file2
- Clone tensorflow
- Go to root of tensorflow
- bazel build tensorflow/examples/image_retraining:retrain
- bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /path/to/root_folder_name  --output_graph /path/output_graph.pb --output_labels /path/output_labels.txt --bottleneck_dir /path/bottleneck
** Training done. **
For testing through bazel,
    bazel build tensorflow/examples/label_image:label_image && \
    bazel-bin/tensorflow/examples/label_image/label_image \
    --graph=/path/output_graph.pb --labels=/path/output_labels.txt \
    --output_layer=final_result \
    --image=/path/to/test/image
For testing through python, change and run this code.
"""

import numpy as np
import tensorflow as tf
import os, time
from shutil import copyfile

MODEL_NAME = 'output_graph.pb'
LABEL_NAME = 'output_labels.txt'
MODEL_VER = 3
WORKSPACE = '/home/ubuntu/tf_model/v%d/' % MODEL_VER
MODEL_PATH = os.path.join(WORKSPACE, MODEL_NAME)
LABEL_PATH = os.path.join(WORKSPACE, LABEL_NAME)

validate_folder = "/home/ubuntu/dataset"

def checkFolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def analyzeIamge(image_path, label_lines):
    answer = None

    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
        return answer

    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
        labels = [str(w).replace("\n", "") for w in label_lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

        answer = labels[top_k[0]]
        score = predictions[top_k[0]]
        return answer, score

if __name__ == '__main__':
    # Creates graph from saved GraphDef.
    create_graph()

    imageList = os.listdir(validate_folder)
    imageList = sorted(imageList, reverse=True)

    mightBeXXX = "mightBeXXX"
    optFolderXXX = os.path.join(WORKSPACE, mightBeXXX)
    checkFolder(optFolderXXX)

    f = open(LABEL_PATH, 'rb')
    label_lines = f.readlines()

    count = 0
    total = len(imageList)
    start = time.time()
    for img in imageList:
        imagePath = os.path.join(validate_folder, img)
        answer, score = analyzeIamge(imagePath, label_lines)
        print "%s : %s" % (img , answer)

        if "xxx" in answer:
            copyfile(imagePath, os.path.join(optFolderXXX, "%s_%.3f.jpg" % (img.split(".")[0], score)))

        count += 1

        if count % 10 == 0:
            print "=" * 100
            spend = time.time() - start
            spendTime = spend / float(count)
            remain = (total - count) * spendTime
            print "[%.2f%%] %d/%d, remain : %d sec" % (count / float(total) * 100, count, total, remain)
            print "=" * 100

    totalSpend = time.time() - start
    print "*" * 100
    print "Spend :%d sec, avg:%.3f sec, total process:%d" % (totalSpend, totalSpend/float(total), total)
