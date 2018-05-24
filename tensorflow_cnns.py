#!/usr/bin/env python

"""
Convolutional neural networks
"""
#~ import numpy as np
#~ from scipy import signal as sg
import tensorflow as tf
#Importing
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image


#~ h = [2, 1, 0]
#~ x = [3, 4, 5]

#~ y = np.convolve(x, h)

#~ # padding
#~ x = [6,2]
#~ h = [1,2,5,4]
#~ y = np.convolve(x, h, 'valid')



#~ I= [[255,   7,  3],
    #~ [212, 240,  4],
    #~ [218, 216, 230],]

#~ g= [[-1,  1],
    #~ [ 2,  3],]

#~ print ('With zero padding \n')
#~ print ('{0} \n'.format(sg.convolve( I, g, 'full')))
#~ # The output is the full discrete linear convolution of the inputs. 
#~ # It will use zero to complete the input matrix

#~ print ('With zero padding_same_ \n')
#~ print ('{0} \n'.format(sg.convolve( I, g, 'same')))
#~ # The output is the full discrete linear convolution of the inputs. 
#~ # It will use zero to complete the input matrix


#~ print ('Without zero padding \n')
#~ print (sg.convolve( I, g, 'valid'))
#~ # The 'valid' argument states that the output consists only of those elements 
#~ #that do not rely on the zero-padding.

input_m = tf.Variable(tf.random_normal([1, 10, 10, 1]))
filter_m = tf.Variable(tf.random_normal([3, 3, 1, 1]))

op = tf.nn.conv2d(input_m, filter_m, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input_m, filter_m, strides=[1, 1, 1, 1], padding='SAME')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    print("Input \n")
    print('{0} \n'.format(input_m.eval()))
    print('{0} \n'.format(filter_m.eval()))
    print("Result/Feature Map with valid positions \n")
    result = sess.run(op)
    print(result)
    print('\n')
    print("Result/Feature Map with padding \n")
    result2 = sess.run(op2)
    print(result2)





    ### Load image of your choice on the notebook
    print("Please type the name of your test image after uploading to \
    your notebook (just drag and grop for upload. Please remember to \
    type the extension of the file. Default: bird.jpg")

    raw= raw_input()

    im = Image.open(raw)  # type here your image's name

    # uses the ITU-R 601-2 Luma transform (there are several 
    # ways to convert an image to grey scale)

    image_gr = im.convert("L")    
    print("\n Original type: %r \n\n" % image_gr)

    # convert image to a matrix with values from 0 to 255 (uint8) 
    arr = np.asarray(image_gr) 
    print("After conversion to numerical representation: \n\n %r" % arr) 
    ### Activating matplotlib for Ipython

    ### Plot image

    imgplot = plt.imshow(arr)
    imgplot.set_cmap('gray')  #you can experiment different colormaps (Greys,winter,autumn)
    print("\n Input image converted to gray scale: \n")
    plt.show(imgplot)

    kernel = np.array([
                            [ 0, 1, 0],
                            [ 1,-4, 1],
                            [ 0, 1, 0],
                                         ]) 

    grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')
    print('GRADIENT MAGNITUDE - Feature map')

    fig, aux = plt.subplots(figsize=(10, 10))
    aux.imshow(np.absolute(grad), cmap='gray')
    
    type(grad)

    grad_biases = np.absolute(grad) + 100

    grad_biases[grad_biases > 255] = 255
    
    print('GRADIENT MAGNITUDE - Feature map')

    fig, aux = plt.subplots(figsize=(10, 10))
    aux.imshow(np.absolute(grad_biases), cmap='gray')
    plt.show(aux)
