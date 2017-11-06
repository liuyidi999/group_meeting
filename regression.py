#!usr/bin/env python
# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np
tf.set_random_seed(1)
np.random.seed(1)
# using numpy to generate data for regression
x_data = np.float32(np.random.rand(2, 100)) #data of independent variables, X1 and X2.
y_data = np.dot([0.100, 0.200], x_data) + 0.300#data of y=0.1*X1+0.2*X2+0.3

# define placeholders receive data from x_data and y_data
x=tf.placeholder(tf.float32,[2,100])
y=tf.placeholder(tf.float32,[100])

# define parameters as variables in tensorflow
with tf.variable_scope('parameters') as scope:
	b = tf.get_variable('threshold',[1])
	W = tf.get_variable('weights',[1,2])

# build up model computation process
y_predict = tf.matmul(W, x) + b

# loss function as Mean Squared Error 
loss_function = tf.reduce_mean(tf.square(tf.subtract(y,y_predict)))

# define training algorithm
train_algorithm = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)

# define initializer
init = tf.global_variables_initializer()
# activate a session
with tf.Session() as sess:
	sess.run(init)#initialize all those variables
	for step in range(300):#train the model every time the data is feeded
		current_loss,nothing=sess.run([loss_function,train_algorithm],feed_dict={x:x_data,y:y_data})
		if step % 20 == 0:
			print ('\n','MSE=',current_loss)
			print('w=',sess.run('parameters/weights:0'),'\n','b=',sess.run('parameters/threshold:0'))


