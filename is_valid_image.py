#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np   # set up Python environment: numpy for numerical routines
import sys           # The caffe module needs to be on the Python path;
import caffe
import os
from os.path import join, getsize 
import datetime 

caffe_root = '/home/tunicorn/software/caffe/'
image_names = caffe_root + 'examples/ukbench/ukbench_names.txt' # a list of names of images
image_dir = '/home/tunicorn/software/caffe/data/ukbench/full/'                # the directory of source images
'''
i = 0
with open(image_names, 'r') as fi:
	while (True):
		line = fi.readline().strip().split()
		if not line:
			break
		i += 1
		imagefile_abs = os.path.join(image_dir, line[0])
		if getsize(imagefile_abs) == 0:
			print imagefile_abs, 'with size 0. No.', i
print 'The total is', i
'''
'''
caffe_root = '/home/tunicorn/software/caffe/'
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'  # defines the structure of the model
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
image_names = caffe_root + sys.argv[1] # a list of names of images
image_dir = sys.argv[2]                # the directory of source images
mean_file = caffe_root + 'data/linmao_temp/linmao_mean.npy'

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

i = 0
with open(image_names, 'r') as fi:
	while (True):
		line = fi.readline().strip().split()
		if not line:
			break
		i += 1
		imagefile_abs = os.path.join(image_dir, line[0])
		try:
			start_tm = datetime.datetime.now()	
			img = caffe.io.load_image(imagefile_abs)
			end_tm = datetime.datetime.now()
			interval = end_tm - start_tm
			print 'caffe.io.load_image:', interval

			start_tm = datetime.datetime.now()
			trans_img = transformer.preprocess('data', img)
			end_tm = datetime.datetime.now()
			interval = end_tm - start_tm
			print 'transformer.preprocess:', interval

			start_tm = datetime.datetime.now()
			one_data = np.asarray([trans_img])
			end_tm = datetime.datetime.now()
			interval = end_tm - start_tm
			print 'np.asarray:', interval
			#one_data = np.asarray([transformer.preprocess('data', caffe.io.load_image(imagefile_abs))])
		except IOError:
			print imagefile_abs, 'cannot open. No.', i
		finally:
			pass
print 'The total is', i
'''
