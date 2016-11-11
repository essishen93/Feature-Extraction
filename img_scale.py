#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np   # set up Python environment: numpy for numerical routines
import sys           # The caffe module needs to be on the Python path;
import caffe
import os
import Image


caffe_root = '/home/tunicorn/software/caffe/'

img_name = caffe_root + 'examples/_temp/0000660e6fd672e1c52840fe3c139d581.jpg'
out_name = caffe_root + 'examples/_temp/small_img.jpg'

im = Image.open(img_name)
out = im.resize(256, 256)
out.save(out_name)

