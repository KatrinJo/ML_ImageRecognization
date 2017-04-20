import numpy as np
import matplotlib.pyplot as plt
import os

caffe_root = '/home/sunshineatnoon/Downloads/caffe/'
import sys
sys.path.insert(0,caffe_root+'python')

import caffe

MODEL_FILE = caffe_root+'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

#cpu模式
caffe.set_mode_cpu()
#定义使用的神经网络模型
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
               mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
               channel_swap=(2,1,0),
               raw_scale=255,
               image_dims=(256, 256))
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

#对目标路径中的图像，遍历并分类
for root,dirs,files in os.walk("/home/sunshineatnoon/Downloads/dogs/dogs/"):
    for file in files:
        #加载要分类的图片
        IMAGE_FILE = os.path.join(root,file).decode('gbk').encode('utf-8');
        input_image = caffe.io.load_image(IMAGE_FILE)
        
        #预测图片类别
        prediction = net.predict([input_image])
        print 'predicted class:',prediction[0].argmax()

        # 输出概率最大的前5个预测结果
        top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
        print labels[top_k]