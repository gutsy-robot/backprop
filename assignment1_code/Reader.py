#!/usr/bin/env python

from mnist import MNIST
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import time
import os


class Reader(object):

	def __init__(self):
		self.data_path = os.path.join(os.path.dirname(__file__), 'data/data')
		self.train_images = None 
		self.test_images = None 
		self.train_labels = None
		self.test_labels = None 
		self.num_train_example = None
		self.num_test_example  = None
		self.train_label_vector = None
		self.test_label_vector = None
		self.num_features = None
		self.class_var = np.array([10.0, 13.0, 17.0, 18.0, 20.0, 27.0, 28.0, 29.0, 31.0])		
		self.num_labels = len(self.class_var)   #we will classify into the nine-letter classes
		self.output_dim = len(self.class_var)




	def read(self):
		mndata = MNIST(self.data_path)

		print "reading data.."
		training_images, training_labels = mndata.load_training()
		print "training ubyte files read..."
		testing_images, testing_labels = mndata.load_testing()
		print "data read successfully from ubyte files"
		train_images = np.array(training_images, dtype=np.float128)
		print "training images numpy array created..."
		train_labels = np.array(training_labels, dtype=np.float128)
		print "train_labels numpy array created..."
		test_images = np.array(testing_images, dtype=np.float128)
		print "test_images numpy array created..."
		test_labels = np.array(testing_labels, dtype=np.float128)
		print "test_labels numpy array created..."

		#just to visualise the training data.
		
		print "conversion to numpy array done..."
		print "images dimensions are.."
		print train_images.shape, test_images.shape
		print "label dimensions are.."
		print train_labels.shape, test_labels.shape

		'''
		first_image = training_images[0]
		first_image = np.array(first_image, dtype='float')
		pixels = first_image.reshape((28, 28))
		plt.imshow(pixels, cmap='gray')
		plt.show()
		'''
		

		#filter out training variables 
		self.train_labels = np.array(filter(lambda x: x in self.class_var, train_labels))
		print "shape of train_labels is %s" % self.train_labels.shape
		self.num_train_example = self.train_labels.shape[0]

		self.test_labels =  np.array(filter(lambda x: x in self.class_var, test_labels))
		self.num_test_example = self.test_labels.shape[0]

		train_indices = [index for index,value in enumerate(train_labels) if value in self.class_var]
		train_images_temp = train_images[train_indices,:]
		self.train_images = train_images_temp/256.0

		test_indices = [index for index,value in enumerate(test_labels) if value in self.class_var]
		test_images_temp = test_images[test_indices,:]
		self.test_images = test_images_temp/256.0
		self.num_features = self.train_images.shape[1]	
		print "calling vectorise output now..."
		self.vectorise_output()
		print "number of features are %s" % str(self.num_features)
		print "number of training examples are %s" % str(self.num_train_example)
		print "number of test examples are %s" % str(self.num_test_example)




	def vectorise_output(self):
     		self.train_label_vector = np.zeros((len(self.train_labels), self.output_dim), dtype=np.float128)
     		self.test_label_vector = np.zeros((len(self.test_labels), self.output_dim), dtype=np.float128)
     		label_indices_list_train = []
     		label_indices_list_test = []

     		for i in range(0,self.num_labels):
     			label_indices_list_train.append([index for index,value in enumerate(self.train_labels) if value in [self.class_var[i]]])
     		for j in range(0,self.num_labels):
     			label_indices_list_test.append([index for index,value in enumerate(self.test_labels) if value in [self.class_var[i]]])

     		for k in range(0,self.num_labels):
     			for t in label_indices_list_train[k]:
     				temp = np.zeros(self.num_labels)
     				temp[k] = 1.0
     				self.train_label_vector[t, :] = temp
			for l in range(0,self.num_labels):
				for u in label_indices_list_test[l]:
					temp = np.zeros(self.num_labels)
					temp[l] = 1.0
					self.test_label_vector[u, :] = temp

			print "output vectors created for training..."
if '__name__' == '__main__':
	obj = Reader()
	obj.read()



