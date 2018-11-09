
'''
train_set_x_flatten shape	(12288, 209)
train_set_y shape	(1, 209)
test_set_x_flatten shape	(12288, 50)
test_set_y shape	(1, 50)
'''

import numpy as np
import cv2

def load_dataset():
	cat01 = cv2.imread('./cats/02.jpg')
	cat01 = cat01[np.newaxis , : , : ,:]
	cat02 = cv2.imread('./cats/04.jpg')
	cat02 = cat02[np.newaxis , : , : ,:]
	ncat01 = cv2.imread('./cats/01.jpg')
	ncat01 = ncat01[np.newaxis , : , : ,:]
	ncat02 = cv2.imread('./cats/03.jpg')
	ncat02 = ncat02[np.newaxis , : , : ,:]
	
	train_set_x_orig = np.concatenate( (cat01 ,ncat01 ,cat02, ncat02) , axis = 0)
	train_set_y = np.array([1,0,1,0]).reshape(1 , 4)
	
	cat03 = cv2.imread('./cats/06.jpg')
	cat03 = cat03[np.newaxis , : , : ,:]
	ncat03 = cv2.imread('./cats/05.jpg')
	ncat03 = ncat03[np.newaxis , : , : ,:]
	
	test_set_x_orig = np.concatenate( (cat03 ,ncat03) , axis = 0)
	
	test_set_y = np.array([1,0]).reshape(1 , 2)
	classes = np.array(["cat","non-cat"])


	return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes 



def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def initialize_parameters(dim):
	
	## dim -- size of the w vector we want (or number of parameters in this case)
	w, b = np.zeros((dim , 1)) , 0
	return 
	
def propagate(w, b, X , Y):
	
	m = X.shape[1]
	
	A = sigmoid( np.dot(w.T , X) + b )
	cost = (-1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A) )
	dw = (1/m) * np.dot(X , (A-Y).T)
	db = (1/m) * np.sum(A - Y)
	
	grads = {"dw" :  dw , 
			 "db" :  db}
	
	return grads , cost


def optimize(w, b, X, Y, alpha, num_iteration):
	
	
	costs =[]

	for i in range(num_iteration):
		
		grads , cost = propagate(w, b, X, Y)
		
		w = w - alpha*grads["dw"]
		b = b - alpha*grads["db"]
		
		costs.append(cost);

		if i % 100 == 0:
			costs.append(cost)

		if i % 100 == 0:
			print("Cost after iteration %i: %f" % (i, cost))

	parameters = {"w": w, "b": b}
	
	return parameters, grads, costs


def predict(w, b, X):

	m = X.shape[1]
	Y_pred = np.zeros((1, m))

	w = w.reshape(X.shape[0],1)
	
	A = sigmoid(np.dot(w.T , X) + b )
	
	Y_pred = np.greater(A, 0.5) * 1
	
	return Y_pred


def logistic_model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5):

	## step 1 :: initialize_parameters
	w, b  = ( np.random.uniform( 0.5,0.8 ,X_train.shape[0]) *0.01).reshape(X_train.shape[0] , 1) , 0
	#w ,b = np.random.randn(X_train.shape[0],1) * 0.01 , 0 
	## step 2 :: train parameters
	parameters , grads , costs = optimize(w, b, X_train, Y_train, learning_rate , num_iterations)

	w = parameters["w"]
	b = parameters["b"]

	## step 3 :: predict
	Y_prediction_test = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)

	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

	d = {"costs": costs,
		 "Y_prediction_test": Y_prediction_test,
		 "Y_prediction_train": Y_prediction_train,
		 "w": w,
		 "b": b,
		 "learning_rate": learning_rate,
		 "num_iterations": num_iterations}

	return d

if __name__ == '__main__':

	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

	m_train = train_set_x_orig.shape[0]   # 4
	m_test = test_set_x_orig.shape[0]     # 2
	num_px =train_set_x_orig.shape[1] 	  # 50

	train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0] , -1).T
	test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0] , -1).T

	train_set_x = train_set_x_flatten /255.
	test_set_x = test_set_x_flatten /255.

	d = logistic_model(train_set_x, train_set_y, test_set_x, test_set_y,
	 					num_iterations = 50, learning_rate = 0.07)