#!/usr/bin/env python

import os
import numpy as np
import struct




def load_data(dataset="training", digits=np.arange(10), path="/home/anish/deep_learning/assignment/", size = 60000):

    
	#define paths to train and test data.
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        print "training images read, now reading labels...."
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
        print "training labels read..."
    elif dataset == "testing":
    	print "reading test images..."
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        print "testing images read...."
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
        print "testing labels read..."
    
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    #open label files
    flbl = open(fname_lbl, 'rb')

    #extract data from label files.
    magic_nr, size = struct.unpack(">II", flbl.read(8))

    lbl = pyarray("b", flbl.read())
    flbl.close()

    #open image file.
    fimg = open(fname_img, 'rb')

    #extract data from training images.
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))

    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = size
    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(N): 
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])\
            .reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    labels = [label[0] for label in labels]
    return images, labels

    print "data reading complete..."

if __name__ == '__main__':
	load_data()