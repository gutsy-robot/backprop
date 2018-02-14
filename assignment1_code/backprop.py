#!/usr/bin/env python

'''
script for reading data from eminist dataset

dependencies: pip install python-mnist

Implementation of a single layered neural network.

Authors: Siddharth Agrawal, Vikas Deep, Harsh Sahu

'''

from mnist import MNIST
import random
import numpy as np
from matplotlib import pyplot as plt
import time
import os





class VanillaBackProp(object):
	def __init__(self):
		print "initialising class variables.."
		self.data_path = fn = os.path.join(os.path.dirname(__file__), 'data')
		self.train_images = None 
		self.test_images = None 
		self.train_labels = None
		self.test_labels = None 
		self.class_var = [1,2,3]										#we will classify into the nine-letter classes.
		#random weight initialisation
		self.weights = None
		self.num_features = None
		self.hidden_layer_dim = 20	
		self.num_hlayers = 1
		self.output_dim = len(self.class_var)
		self.num_train_example = None
		self.num_test_example  = None
		#represents the output of the current feed forward prop
		self.train_label_vector = None
		self.test_label_vector = None
		self.z2 = None
		self.z3 = None
		self.a2 = None
		self.a3 = None
		self.b1 = None
		self.b2 = None

		#regularisation parameter for L2
		self.reg_lambda = 0.01

		#specify learning rate
		self.epsilon = 0.001

		#number of passes while training
		self.num_passes = 50
		self.predicted_outputs = None
		self.predicted_output_labels = None






	def read(self):
		mndata = MNIST(self.data_path)

		print "reading data.."
		training_images, training_labels = mndata.load_training()
		testing_images, testing_labels = mndata.load_testing()
		train_images = np.array(training_images, dtype=np.float128)
		train_labels = np.array(training_labels, dtype=np.float128)
		test_images = np.array(testing_images, dtype=np.float128)
		test_labels = np.array(testing_labels, dtype=np.float128)


		
		#just to visualise the training data.
		'''
		print "conversion to numpy array done..."
		print "images dimensions are.."
		print train_images.shape, test_images.shape
		print "label dimensions are.."
		print train_labels.shape, test_labels.shape
		
		first_image = training_images[0]
		first_image = np.array(first_image, dtype='float')
		pixels = first_image.reshape((28, 28))
		plt.imshow(pixels, cmap='gray')
		plt.show()
		'''
		

		#filter out training variables 
		self.train_labels = np.array(filter(lambda x: x in self.class_var, train_labels))
		self.num_train_example = self.train_labels.shape[0]

		self.test_labels =  np.array(filter(lambda x: x in self.class_var, test_labels))
		self.num_test_example = self.test_labels.shape[0]

		train_indices = [index for index,value in enumerate(train_labels) if value in self.class_var]
		train_images_temp = train_images[train_indices,:]
		self.train_images = train_images_temp/256.0

		test_indices = [index for index,value in enumerate(test_labels) if value in self.class_var]
		test_images_temp = test_images[test_indices,:]
		self.test_images = test_images_temp/256.0
		#print "dimension of test images..."
		#print self.test_images.shape[0], self.test_images.shape[1]
		#print "printing test images.."


		self.num_features = self.train_images.shape[1]	
		self.vectorise_output()
		print "number of features are %s" % str(self.num_features)
		print "number of training examples are %s" % str(self.num_train_example)
		print "number of test examples are %s" % str(self.num_test_example)
		
	def sigmoidDerivative(self,z):
		return np.exp(-z)/((1+np.exp(-z))**2)
	

	def initialise_weights(self):
		self.w1 = 2*np.random.random((self.num_features, self.hidden_layer_dim)) - 1
		self.w2 = 2*np.random.random((self.hidden_layer_dim, self.output_dim)) - 1
		self.b1 = np.zeros((1, self.hidden_layer_dim), dtype=np.float128)
		self.b2 = np.zeros((1, self.output_dim), dtype=np.float128)

	#implemented only for the sigmoid activation for now.

	def learn(self, reg=None, cost_type='MSE'):
		print "learning weights..."
		for i in xrange(0,self.num_passes):
			self.z2 = self.train_images.dot(self.w1) + self.b1
			self.a2 =  1/(1 + np.exp(-self.z2))	
			#print self.current_output[1]

			self.z3 = self.a2.dot(self.w2) + self.b2
			self.a3 = 1/(1 + np.exp(-self.z3))
			#backprop

			if cost_type=='MSE':

				delta3 = np.multiply(-(self.train_label_vector-self.a3), self.sigmoidDerivative(self.z3))
				db2 = np.sum(delta3, axis=0, keepdims=True)
				dJdW2 = np.dot(self.a2.T, delta3)

				delta2 = np.dot(delta3, self.w2.T)*self.sigmoidDerivative(self.z2)
				db1 = np.sum(delta2, axis=0)
				dJdW1 = np.dot(self.train_images.T, delta2)



			elif cost_type== 'cross' :

				delta3 = -(self.train_label_vector-self.a3)

				delta2 = np.dot(delta3, self.w2.T)*self.sigmoidDerivative(self.z2)
				db2 = np.sum(delta3, axis=0, keepdims=True)
				dJdW1 = np.dot(self.train_images.T, delta2)
			

			#do L2 regularisation.
			if reg=='L2':
				print"L2 regularisation is set to true.."
				dJdW2 += self.reg_lambda * self.w2
				dJdW1 += self.reg_lambda * self.w1
	 		if reg=='L1':
	 			print"L1 regularisation is set to true..."
	 			dJdW2 += self.reg_lambda * np.sign(self.w2)
				dJdW1 += self.reg_lambda * np.sign(self.w1)
	        

			self.w1 += -self.epsilon * dJdW1
			self.b1 += -self.epsilon * db1
			self.w2 += -self.epsilon * dJdW2
			self.b2 += -self.epsilon * db2
			cost = self.computeCost(cost_type,reg)
			print "cost is after iteration: %s" % str(i)
			print cost
			plt.scatter(i,cost)
		plt.show()

	def vectorise_output(self):
     		self.train_label_vector = np.zeros((len(self.train_labels), 3), dtype=np.float128)
     		self.test_label_vector = np.zeros((len(self.test_labels), 3), dtype=np.float128)

     		label1 = [index for index,value in enumerate(self.train_labels) if value in [1]]
     		label2 = [index for index,value in enumerate(self.train_labels) if value in [2]]
     		label3 = [index for index,value in enumerate(self.train_labels) if value in [3]]
     		for x in label1:
				self.train_label_vector[x, :] = [1.0, 0.0, 0.0]
     		for y in label2:
				self.train_label_vector[y, :] = [0.0, 1.0, 0.0]
     		
     		for z in label3:
				self.train_label_vector[z, :] = [0.0, 0.0, 1.0]
		print "dimension of train label vector is..."
		print self.train_label_vector.shape[0], self.train_label_vector.shape[1]

		label1_test = [index for index,value in enumerate(self.test_labels) if value in [1]]
     		label2_test = [index for index,value in enumerate(self.test_labels) if value in [2]]
     		label3_test = [index for index,value in enumerate(self.test_labels) if value in [3]]
     		for x in label1_test:
				self.test_label_vector[x, :] = [1.0, 0.0, 0.0]
     		for y in label2_test:
				self.test_label_vector[y, :] = [0.0, 1.0, 0.0]
     		
     		for z in label3_test:
				self.test_label_vector[z, :] = [0.0, 0.0, 1.0]
		
     		
     		print "output vectors created for training..."

	def predict(self):
	        print "Predicting test data..."
	        #print "dimension of test image is..."
	        #print self.test_images.shape[0], self.test_images.shape[1]
	        z2 = self.test_images.dot(self.w1) + self.b1
	        a2 =  1/(1 + np.exp(-z2))	
	        z3 = a2.dot(self.w2) + self.b2
	        a3 = 1/(1 + np.exp(-z3))
	        self.predicted_outputs = a3
	        #print "dimension of the predicted outputs are.."
	        #print self.predicted_outputs.shape[0], self.predicted_outputs.shape[1]
	        labels = np.argmax(self.predicted_outputs, axis=1)
	        self.predicted_output_labels = 1 + labels
	        #print self.predicted_output_labels
	       
	        #text_file2 = open("Output2.txt", "w")
	        #text_file2.write("outputs are: %s" % str(self.predicted_output_labels))
	        #text_file2.close()
	        '''
	        for i in range(0,300):
	        	text_file2.write(str(self.predicted_output_labels[i]))
	        text_file2.close()
	        '''
	        print "predicting done..."


	#function for calculating MSE Loss.
	#def calculate_loss(self):

	def computeCost(self, cost_type='MSE', reg=None):
		if cost_type=='MSE':
			# print "cost is divided by"
			#print self.train_label_vector.shape[0]
			J = (sum(sum((self.train_label_vector - self.a3)**2)))/self.train_label_vector.shape[0]
			

			if reg == None:
				return J
			elif reg == 'L2':
				J += self.reg_lambda*()/(2*self.train_label_vector.shape[0]) + self.reg_lambda*()/(2*self.train_label_vector.shape[0])

			elif reg== 'L1':
				J += self.reg_lambda*()/(2*self.train_label_vector.shape[0]) + self.reg_lambda*()/(2*self.train_label_vector.shape[0])


		elif cost_type=='cross':
			print "hi"

	def calculate_accuracy(self):
		print "calculating accuracy"
		#print self.predicted_output_labels.shape[0]
		#print self.test_labels.shape[0]
		num_correct = np.count_nonzero(self.predicted_output_labels==self.test_labels)
		print num_correct
		print self.num_train_example
		#print float(num_correct)/float(self.test_labels.shape[0])
		accuracy = (float(num_correct)/float(self.num_test_example))*100
		print "accuracy is: %s" % str(accuracy)





if __name__ == '__main__':
	back_prop = VanillaBackProp()
	back_prop.read()
	back_prop.initialise_weights()
	back_prop.learn()
	back_prop.predict()
	back_prop.calculate_accuracy()
	
