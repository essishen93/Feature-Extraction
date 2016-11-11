#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np   # set up Python environment: numpy for numerical routines
import sys           # The caffe module needs to be on the Python path;
import caffe
import os
import datetime 

caffe_root = '/home/tunicorn/software/caffe/examples/_temp/'
fea_file = caffe_root + 'LinMao_'    # the output binary file of feature vectors
postfix = '.txt'
#fea_file = caffe_root + sys.argv[1]    # the output binary file of feature vectors
#postfix = sys.argv[2]

file_batch_num = 512

for x in range(2048):
	for y in range(128):  
		num = int(x/file_batch_num) + 1
		one_fea_file = fea_file + str(num) + postfix
		with open(one_fea_file, 'w') as f:
			f.write(str(num))
			f.write(str(x))