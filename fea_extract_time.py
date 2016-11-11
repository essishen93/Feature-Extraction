#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np   # set up Python environment: numpy for numerical routines
import sys           # The caffe module needs to be on the Python path;
import caffe
import os
import datetime 
#import skimage
#from skimage import data, io

# Variable Definition
caffe_root = '/home/tunicorn/software/caffe/'
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'  # defines the structure of the model
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'  # contains the trained weights
mean_file = caffe_root + 'data/linmao_temp/linmao_mean.npy'                  # a file with 3 mean values of each channel
image_names = caffe_root + sys.argv[1] # a list of names of images
image_dir = sys.argv[2]                # the directory of source images
fea_file = caffe_root + sys.argv[3]    # the output binary file of feature vectors
total_num = int(sys.argv[4])           # the total number of images
batch_num = int(sys.argv[5])           # the batch size


# Set up the net
sys.path.insert(0, caffe_root + 'python')    # Set Caffe to CPU mode and load the net from disk.
caffe.set_mode_cpu()
net = caffe.Net(model_def, model_weights, caffe.TEST) 

# Create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))                          # move image channels to outermost dimension
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)                              # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))                       # swap channels from RGB to BGR

# set the size of the input 
net.blobs['data'].reshape(batch_num, 3, 227, 227)  
batch = np.zeros((batch_num, 3, 227, 227),dtype=np.float)

loop_num = total_num/batch_num
with open(image_names, 'r') as fi:
	for x in range(loop_num):
		start_batch_time = datetime.datetime.now() 
		starttime = datetime.datetime.now() 
		for y in range(batch_num):   # creat one batch to do forward
			line = fi.readline().strip().split()
			if not line:
				break
			start_small = datetime.datetime.now()
			imagefile_abs = os.path.join(image_dir, line[0])
			end_small = datetime.datetime.now()
			interval = end_small - start_small
			print 'read name: ', interval

			start_small = datetime.datetime.now()
			one_data = np.asarray([transformer.preprocess('data', caffe.io.load_image(imagefile_abs))])
			#one_data = np.asarray([transformer.preprocess('data', skimage.io.imread(imagefile_abs)/255.0)])
			end_small = datetime.datetime.now()
			interval = end_small - start_small
			print 'read in image: ', interval

			start_small = datetime.datetime.now() 
			batch [y, :, :, :] = one_data
			end_small = datetime.datetime.now()
			interval = end_small - start_small
			print 'store in blob: ', interval
		endtime = datetime.datetime.now() 
		interval = endtime - starttime 
		print 'Created one batch. [doing forward...]', 'The time is ', interval

		starttime = datetime.datetime.now() 
		out = net.forward_all( data = batch )
		endtime = datetime.datetime.now() 
		interval = endtime - starttime
		print 'Done forward. [writing file...]', 'The time is ', interval

		starttime = datetime.datetime.now()
		with open(fea_file, 'a') as f:
				f.write(net.blobs['fc7'].data)
		endtime = datetime.datetime.now() 
		interval = endtime - starttime
		print 'Written fea_vec of one batch: ', x+1, 'The time is ', interval
		end_batch_time = datetime.datetime.now() 
		interval = end_batch_time - start_batch_time
		print 'The time of processing one batch (', batch_num, ') is', interval
		print