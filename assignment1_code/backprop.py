#!/usr/bin/env python

'''
script for doing backprop with different activation functions on mnist dataset.

dependencies: pip install python-mnist


Authors: Siddharth Agrawal, Vikas Deep, Harsh Sahu

learning rate required with cross-entropy almost one-tenth of that with MSE.

'''

from mnist import MNIST
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import time
import os
from Reader import Reader
from pylab import figure, axes, pie, title, show






class VanillaBackProp(object):
	def __init__(self, layers=2, h_dim=50, epoc=10, reg_lambda=0.001, learn_rate=0.0001, activation='sigmoid', cost_type='MSE', reg_type=None, dropout=False, reader=None ):
		print "initialising class variables.."
		self.data_path = os.path.join(os.path.dirname(__file__), 'data/data')
		self.fig_path = os.path.join(os.path.dirname(__file__), 'plots')
		self.train_images = reader.train_images
		self.test_images = reader.test_images
		self.train_labels = reader.train_labels
		self.test_labels = reader.test_labels 
		self.class_var = reader.class_var										#we will classify into the nine-letter classes
		self.num_features = reader.num_features
		self.hidden_layer_dim = h_dim
		self.num_hlayers = layers
		
		self.num_train_example = reader.num_train_example
		self.num_test_example  = reader.num_test_example

		self.class_var_codes = {'A':65, 'D':68, 'H':72, 'I':73, 'K':75, 'R':82, 'S':83, 'T':84, 'V':86}
		#self.class_var = [5.0, 8.0, 2.0, 3.0, 15.0, 32.0, 33.0, 28.0, 12.0 ]
		#self.class_output_map = {'A':1, 'D':2, 'H':3, 'I':4 'K':5, 'R':6, 'S':7, 'T':8, 'V':9}
		self.output_dim = len(reader.class_var)
		self.train_label_vector = reader.train_label_vector
		self.test_label_vector = reader.test_label_vector
		self.A = None
		self.Z = None
		self.model_weights = None
		self.model_bias = None

		self.num_labels = len(reader.class_var)
		self.reg_lambda = reg_lambda
		self.epsilon = learn_rate

		#number of passes while training
		self.num_passes = epoc
		self.cost_type = cost_type
		self.reg_type = reg_type
		self.activation = activation

		self.predicted_outputs = None
		self.predicted_labels = None
		self.dropout = dropout

		self.train_error_data = []
		self.test_error_data = []
		self.accuracy_data = []
		print "class variables initialised.."

		
	def sigmoidDerivative(self,z):
		return np.exp(-z)/((1+np.exp(-z))**2)

	def reluDerivative(self,z):
		return np.greater(z,0).astype(float)

	def tanh_deriv(self,z):
		return 1.0 - np.tanh(z)**2
	

	def initialise_weights(self):
		weights = []
		bias = []
		w1 = 2*np.random.random((self.num_features, self.hidden_layer_dim)) - 1
		w_last = 2*np.random.random((self.hidden_layer_dim, self.num_labels)) - 1
		b1 = np.zeros((1, self.hidden_layer_dim))#, dtype=np.float128)
		b_last = np.zeros((1, self.output_dim)) #, dtype=np.float128)
		weights.append(w1)
		bias.append(b1)

		for i in range(0,self.num_hlayers-1):
			w = 2*np.random.random((self.hidden_layer_dim, self.hidden_layer_dim)) - 1
			b = np.zeros((1, self.hidden_layer_dim)) #, dtype=np.float128)
			weights.append(w)
			bias.append(b)

		weights.append(w_last)
		bias.append(b_last)

		self.model_weights = weights
		self.model_bias = (bias)

		

	def learn_model(self):
		print "learning weights..."
		print "no of features are %s" % str(self.num_features)
		#new_stuff
		

		for i in range(0,self.num_passes):
			Z = []
			A = [self.train_images]
			dbias = []
			DJDW = []
			k = -2
			dropout_u = []
			if self.dropout == True:
				for x in range(0, self.num_hlayers):
					u = np.random.binomial(1, 0.5, (self.num_train_example ,self.hidden_layer_dim)) / 0.5
					dropout_u.append(u)

			for j in range(0, self.num_hlayers+1):
				z = A[-1].dot(self.model_weights[j]) + self.model_bias[j]

				Z.append(z)
				if self.activation == 'sigmoid':
					a = 1/(1 + np.exp(-z))
				elif self.activation == 'tanh':
					if j!=self.num_hlayers:
						a = np.tanh(z)
					else:
						a= 1/(1 + np.exp(-z))
				if j!=0 and j!= self.num_hlayers:
					if self.dropout==True:
						
						a *= dropout_u[j-1]
						


				A.append(a)
			self.A = A
			self.Z = Z
			


			print "Forward propagation compete, starting backprop"
			if self.activation == 'sigmoid':
				if self.cost_type=='MSE':
					delta = np.multiply(-(self.train_label_vector-self.A[-1]), self.sigmoidDerivative(self.Z[-1]))
				elif self.cost_type=='cross':
					delta = -(self.train_label_vector-self.A[-1])

				
				db_last = np.sum(delta, axis=0, keepdims=True)
				dbias.append(db_last)
				dJdW = np.dot(self.A[k].T, delta)
				DJDW.append(dJdW)
				k -= 1
				for j in range(0,self.num_hlayers):
					
					#dJdW2 = np.dot(self.a2.T, delta)
					delta_b =  np.dot(delta, self.model_bias[k+2].T)*self.sigmoidDerivative(self.Z[k+1])
					db = np.sum(delta_b, axis=0)
					dbias.append(db)
					delta = np.dot(delta, self.model_weights[k+2].T)*self.sigmoidDerivative(self.Z[k+1])
					
					dJdW = np.dot(self.A[k].T, delta)
					DJDW.append(dJdW)
					k -=1
					#print "one iteration done.."

			elif self.activation == 'tanh':
				if self.cost_type=='MSE':
					delta = np.multiply(-(self.train_label_vector-A[-1]), self.sigmoidDerivative(self.Z[-1]))
					db_last = np.sum(delta, axis=0, keepdims=True)
					dbias.append(db_last)
					dJdW = np.dot(self.A[k].T, delta)
					DJDW.append(dJdW)
					k -= 1
					for j in range(0,self.num_hlayers):
					
						#dJdW2 = np.dot(self.a2.T, delta)
						delta_b =  np.dot(delta, self.model_bias[k+2].T)*self.tanh_deriv(self.Z[k+1])
						db = np.sum(delta_b, axis=0)
						dbias.append(db)
						delta = np.dot(delta, self.model_weights[k+2].T)*self.tanh_deriv(self.Z[k+1])
						
						dJdW = np.dot(self.A[k].T, delta)
						DJDW.append(dJdW)
						k -=1
						#print "one iteration done.."


				elif self.cost_type=='cross':
					print "tanh not to be used with cross-entropy.."

				

			db = np.array(dbias)
			db =db[::-1]
			DJDW_v = np.array(DJDW)
			DJDW_v = DJDW_v[::-1]
			

			#do L2 regularisation.
			if self.reg_type=='L2':
				print"L2 regularisation is set to true.."
				for t in range(0,len(self.model_weights)):
					DJDW_v[t] += self.reg_lambda * self.model_weights[t]
				
	 		if self.reg_type =='L1':
	 			print"L1 regularisation is set to true..."
	 			for u in range(0, len(self.model_weights)):
	 				DJDW_v[u] += self.reg_lambda * np.sign(self.model_weights[u])
				
	        
			for k in range(0, len(self.model_weights)):
				self.model_weights[k] += -self.epsilon * DJDW_v[k]
				self.model_bias[k] += -self.epsilon * db[k]
			self.predict()
			accuracy = self.calculate_accuracy()
			train_cost, test_cost = self.computeCost()
			print "training cost after iteration: %s" % str(i+1)
			print train_cost
			print "test cost after iteration: %s" % str(i+1)
			print test_cost
			#plt.scatter((i+1),train_cost)
			self.train_error_data.append(((i+1),train_cost))
			#plt.scatter((i+1), test_cost)
			self.test_error_data.append(((i+1),test_cost))
			self.accuracy_data.append(((i+1),accuracy))
		#plt.show()


		
	def predict(self):
		print "Predicting test data..."
		Z = []
		A = [self.test_images]
		for j in range(0, self.num_hlayers+1):
			z = A[-1].dot(self.model_weights[j]) + self.model_bias[j]
			Z.append(z)
			if self.activation == 'sigmoid':
				a = 1/(1 + np.exp(-z))
			elif self.activation == 'tanh':
				a = np.tanh(z)
			A.append(a)
		self.predicted_outputs = np.array(A[-1])
		labels = np.argmax(self.predicted_outputs, axis=1)
		print "labels are %s" % str(labels.shape)
		self.predicted_labels = self.class_var[labels]
		print "predicted_labels are..."
		print self.predicted_labels
		print "predicting done...."


	#returns cost associated with training and test data, given the current model.
	def computeCost(self):
		if self.cost_type=='MSE':
			print "calculating MSE Loss"

			J_train = (sum(sum((self.train_label_vector - self.A[-1])**2)))/(self.num_train_example)
			J_test = (sum(sum((self.test_label_vector - self.predicted_outputs)**2)))/(self.num_test_example)

		elif self.cost_type=='cross':
			print "calculating cross-entropy loss.."

			J_train = -sum(sum(self.train_label_vector*np.log(self.A[-1]) + (1-self.train_label_vector)*np.log(1-self.A[-1])))/(2*self.num_train_example)
			J_test = -sum(sum(self.test_label_vector*np.log(self.predicted_outputs) + (1-self.test_label_vector)*np.log(1-self.predicted_outputs)))/(2*self.num_test_example)

			

		if self.reg_type == None:
			return J_train, J_test
		elif self.reg_type == 'L2':
			for i in range(0, len(self.model_weights)):
				print "cost using L2"
				J_train += self.reg_lambda*(sum(sum(self.model_weights[i]**2)))/(2*self.num_train_example) 
				J_test += self.reg_lambda*(sum(sum(self.model_weights[i]**2)))/(2*self.num_test_example)
				return J_train, J_test
		elif self.reg_type== 'L1':
			print "cost using L1"
			for i in range(0, len(self.model_weights)):
				J_train += self.reg_lambda*(sum(sum(abs(self.model_weights[i]))))/(2*self.num_train_example) 
				J_test += self.reg_lambda*(sum(sum(abs(self.model_weights[i]))))/(2*self.num_test_example) 
				return J_train, J_test
		

	def calculate_accuracy(self):
		print "calculating accuracy"
		print self.predicted_labels.shape
		num_correct = np.count_nonzero(self.predicted_labels==self.test_labels)
		print num_correct
		print self.test_labels
		accuracy = (float(num_correct)/float(self.num_test_example))*100
		print "accuracy is: %s" % str(accuracy)
		return accuracy

	def getData(self):
		test_error_data = np.array(self.test_error_data)
		train_error_data = np.array(self.train_error_data)
		accuracy_data = np.array(self.accuracy_data)
		print "numpy arrays created from raw data..."
		return train_error_data, test_error_data, accuracy_data


if __name__ == '__main__':
	reader = Reader()
	reader.read()

	
	for w in range(0,1):
		if w ==0:
			back_prop = VanillaBackProp(1, 50, 10, 0.0001, 0.0002,'sigmoid', 'MSE', 'L1', False, reader)
		elif w==1:
			back_prop = VanillaBackProp(2, 50, 10, 0.0001, 0.0002,'sigmoid', 'MSE', 'L1', False, reader)
		elif w==2:
			back_prop = VanillaBackProp(1, 50, 50, 0.0001, 0.0002,'sigmoid', 'MSE', 'L2', False, reader)
		elif w==3:
			back_prop = VanillaBackProp(2, 50, 50, 0.0001, 0.0002,'sigmoid', 'MSE', 'L2', False, reader)
		elif w==4:
			back_prop = VanillaBackProp(1, 50, 50, 0.0001, 0.0002,'sigmoid', 'MSE', None, True, reader)
		elif w==5:
			back_prop = VanillaBackProp(2, 50, 50, 0.0001, 0.0002,'sigmoid', 'MSE', None, True, reader)		
		elif w==6:
			back_prop = VanillaBackProp(1, 50, 50, 0.0001, 0.0002,'sigmoid', 'MSE', None, False, reader)
		elif w==7:
			back_prop = VanillaBackProp(2, 50, 50, 0.0001, 0.0002,'sigmoid', 'MSE', None, False, reader)

		if w ==8:
			back_prop = VanillaBackProp(1, 50, 50, 0.00001, 0.0002,'sigmoid', 'cross', 'L1', False, reader)
		elif w==9:
			back_prop = VanillaBackProp(2, 50, 50, 0.00001, 0.0002,'sigmoid', 'cross', 'L1', False, reader)
		elif w==10:
			back_prop = VanillaBackProp(1, 50, 50, 0.00001, 0.0002,'sigmoid', 'cross', 'L2', False, reader)
		elif w==11:
			back_prop = VanillaBackProp(2, 50, 50, 0.00001, 0.0002,'sigmoid', 'cross', 'L2', False, reader)
		elif w==12:
			back_prop = VanillaBackProp(1, 50, 50, 0.00001, 0.0002,'sigmoid', 'cross', None, True, reader)
		elif w==13:
			back_prop = VanillaBackProp(2, 50, 50, 0.00001, 0.0002,'sigmoid', 'cross', None, True, reader)		
		elif w==14:
			back_prop = VanillaBackProp(1, 50, 50, 0.00001, 0.0002,'sigmoid', 'cross', None, False, reader)
		elif w==15:
			back_prop = VanillaBackProp(2, 50, 50, 0.00001, 0.0002,'sigmoid', 'cross', None, False, reader)

		if w ==16:
			back_prop = VanillaBackProp(1, 50, 50, 0.0001, 0.0002,'tanh', 'MSE', 'L1', False, reader)
		elif w==17:
			back_prop = VanillaBackProp(2, 50, 50, 0.0001, 0.0002,'tanh', 'MSE', 'L1', False, reader)
		elif w==18:
			back_prop = VanillaBackProp(1, 50, 50, 0.0001, 0.0002,'tanh', 'MSE', 'L2', False, reader)
		elif w==19:
			back_prop = VanillaBackProp(2, 50, 50, 0.0001, 0.0002,'tanh', 'MSE', 'L2', False, reader)
		elif w==20:
			back_prop = VanillaBackProp(1, 50, 50, 0.0001, 0.0002,'tanh', 'MSE', None, True, reader)
		elif w==21:
			back_prop = VanillaBackProp(2, 50, 50, 0.0001, 0.0002,'tanh', 'MSE', None, True, reader)		
		elif w==22:
			back_prop = VanillaBackProp(1, 50, 50, 0.0001, 0.0002,'tanh', 'MSE', None, False, reader)
		elif w==23:
			back_prop = VanillaBackProp(2, 50, 50, 0.0001, 0.0002,'tanh', 'MSE', None, False, reader)		
				


		back_prop.initialise_weights()
		back_prop.learn_model()
		train_error, test_error, accuracy = back_prop.getData()
		print "shape of train_error with iteration numpy array..."
		print train_error.shape
		iterations = train_error[:,0]
		train_error = train_error[:,1]
		test_error = test_error[:,1]
		plt.scatter(iterations, train_error, color='red')
		plt.plot(iterations, train_error, color='red' )
		plt.scatter(iterations, test_error, color='blue')
		plt.plot(iterations, test_error, color='blue')
		training_error = mpatches.Patch(color='red', label='Training Error')
		testing_error = mpatches.Patch(color='blue', label='Test Error')
		plt.legend(handles=[training_error, testing_error])
		plt.xlabel('Epochs')
		plt.ylabel('Error')
		if back_prop.dropout:
			plt.title('acc: '+str(int(back_prop.accuracy_data[-1][1]))+' cost: '+str(back_prop.cost_type)+' activation: '+str(back_prop.activation)+' Reg: '+str(back_prop.reg_type)+' layers: '+str(back_prop.num_hlayers)+' h_dim: '+str(back_prop.hidden_layer_dim)+' dropout')
		else:
			plt.title('acc: '+str(int(back_prop.accuracy_data[-1][1]))+ 'cost: '+str(back_prop.cost_type)+' activation: '+str(back_prop.activation)+' Reg: '+str(back_prop.reg_type)+' layers: '+str(back_prop.num_hlayers)+' h_dim: '+str(back_prop.hidden_layer_dim))
		plt.savefig(str(back_prop.fig_path)+'/image'+str(w)+'.png')
		plt.close()
		#plt.show()




	
