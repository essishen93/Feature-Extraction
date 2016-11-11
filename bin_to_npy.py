import PIL
import Image
import sys
import time
import os
import numpy as np
from matplotlib import pyplot as plt 

start = time.time()

# Make sure that caffe is on the python path
caffe_root = '/home/tunicorn/software/caffe/' 
sys.path.insert(0, caffe_root + 'python')

import caffe
# "source" is the binary file converted by the command shell 
# "mean_npy" is the binary file with python format converted from "source"
source = caffe_root + sys.argv[1]
mean_npy = caffe_root + sys.argv[2]

# BlobProto object
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( source , 'rb' ).read()
# parsing source data
blob.ParseFromString(data)
# convert to npy format
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
# save the converted result
with open(mean_npy, 'w') as f:
	np.save(f , out)
# display the mean values
mu = np.load(mean_npy).mean(1).mean(1)
print 'mean-subtracted values:', zip('BGR', mu)