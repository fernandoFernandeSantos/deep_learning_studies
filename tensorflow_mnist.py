#!/usr/bin/env python

"""
Hello world for tensor flow
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

sess = tf.InteractiveSession()


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 10 # number of possible classifications for the problem

x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

x_image = tf.reshape(x, [-1,28,28,1])  

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs


##  Conv layer 1
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

# Relu layer
h_conv1 = tf.nn.relu(convolve1)

#Maxpool
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

## Conv layer 2
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+ b_conv2

# Relu
h_conv2 = tf.nn.relu(convolve2)

## Maxpool
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

## Fully conected layer
layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs

fcl=tf.matmul(layer2_matrix, W_fc1) + b_fc1

# Relu
h_fc1 = tf.nn.relu(fcl)

# Dropout
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

fc=tf.matmul(layer_drop, W_fc2) + b_fc2
y_CNN= tf.nn.softmax(fc)


layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run session
sess.run(tf.global_variables_initializer())

#Train
for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

kernels = sess.run(tf.reshape(tf.transpose(W_conv1, perm=[2, 3, 0,1]),[32,-1]))

sess.close()

#~ mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#~ sess = tf.InteractiveSession()
#~ x  = tf.placeholder(tf.float32, shape=[None, 784])
#~ y_ = tf.placeholder(tf.float32, shape=[None, 10])

#~ # Weight tensor
#~ W = tf.Variable(tf.zeros([784,10],tf.float32))
#~ # Bias tensor
#~ b = tf.Variable(tf.zeros([10],tf.float32))

#~ # run the op initialize_all_variables using an interactive session
#~ sess.run(tf.global_variables_initializer())

#~ #mathematical operation to add weights and biases to the inputs
#~ tf.matmul(x,W) + b

#~ y = tf.nn.softmax(tf.matmul(x,W) + b)

#~ cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#~ train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#~ #Load 50 training examples for each training iteration   
#~ for i in range(1000):
    #~ batch = mnist.train.next_batch(50)
    #~ train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    
    
#~ correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#~ accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#~ acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
#~ print("The final accuracy for the simple ANN model is: {} % ".format(acc) )

#~ sess.close() #finish the session
