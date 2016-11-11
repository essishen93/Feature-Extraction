#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np   # set up Python environment: numpy for numerical routines
import sys           # The caffe module needs to be on the Python path;
import caffe
import os
import datetime 
from sklearn import preprocessing

# Variable Definition
caffe_root = '/home/tunicorn/software/caffe/'
# model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'  # defines the structure of the model
# model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'  # contains the trained weights
model_def = caffe_root + 'models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers_deploy.prototxt'  
model_weights = caffe_root + 'models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel'  
mean_file = caffe_root + 'data/linmao_temp/linmao_mean_12000.npy'                  # a file with 3 mean values of each channel
image_names = caffe_root + sys.argv[1] # a list of names of images
image_dir = sys.argv[2]                # the directory of source images
fea_file = caffe_root + sys.argv[3]    # the output binary file of feature vectors
total_num = int(sys.argv[4])           # the total number of images
batch_num = int(sys.argv[5])           # the batch size
# file_batch_num = int(65536/batch_num)  # the number of batch in one feature file(1GB)
# postfix = '.bin'
gpu_id = 0


# Set up the net
sys.path.insert(0, caffe_root + 'python')    # Set Caffe to CPU mode and load the net from disk.
caffe.set_mode_gpu()
caffe.set_device(gpu_id);
net = caffe.Net(model_def, model_weights, caffe.TEST) 

# Create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))                          # move image channels to outermost dimension
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)                              # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))                       # swap channels from RGB to BGR

# set the size of the input 
net.blobs['data'].reshape(batch_num, 3, 224, 224)  
batch = np.zeros((batch_num, 3, 224, 224),dtype=np.float)

loop_num = total_num/batch_num
start_time = datetime.datetime.now() 
with open(image_names, 'r') as fi:
	for x in range(loop_num):
		start_batch_time = datetime.datetime.now() 
		for y in range(batch_num):   # creat one batch to do forward
			line = fi.readline().strip().split()
			if not line:
				break
			imagefile_abs = os.path.join(image_dir, line[0])
			one_data = np.asarray([transformer.preprocess('data', caffe.io.load_image(imagefile_abs))])
			batch [y, :, :, :] = one_data
		print 'Created one batch. [doing forward...]'
		end_crt_time = datetime.datetime.now()
		interval = end_crt_time - start_batch_time
		print 'Time is', interval
		out = net.forward_all( data = batch )
		print 'Done forward. [normalization...]'
		end_for_time = datetime.datetime.now()
		interval = end_for_time - end_crt_time
		print 'Time is', interval
		#num = int(x/file_batch_num) + 1
		#one_fea_file = fea_file + str(num) + postfix
		im_features = net.blobs['fc7'].data
		im_features = preprocessing.normalize(im_features, norm='l2')
		print 'Done normalization. [writing file...]'
		with open(fea_file, 'a') as f:
				f.write(im_features)
		print 'Written fea_vec of one batch: ', x+1
		end_batch_time = datetime.datetime.now() 
		interval = end_batch_time - end_for_time
		print 'Time is', interval
		interval = end_batch_time - start_batch_time
		print 'The time of processing one batch (', batch_num, ') is', interval
		print
end_time = datetime.datetime.now() 
interval = end_time - start_time
print 'The total time (', total_num, ') is', interval
