#!/usr/bin/env python

"""
Hello world for tensor flow
"""
import tensorflow as tf


a = tf.constant([2])
b = tf.constant([3])

#c = a + b is also a way to define the sum of the terms
c = tf.add(a, b)

with tf.Session() as session:

    result = session.run(c)

    print(result)

    scalar = tf.constant([2])
    vector = tf.constant([3,4,5])
    matrix = tf.constant([[34,5,6,4], [2,3,4,2], [3,4,5,3], [3,4,5,3]])
    tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
    result = session.run(scalar)
    #~ print "Scalar (1 entry):\n %s \n" % result
    #~ result = session.run(vector)
    #~ print "Vector (3 entries) :\n %s \n" % result
    #~ result = session.run(matrix)
    #~ print "Matrix (3x3 entries):\n %s \n" % result
    #~ result = session.run(tensor)
    #~ print "Tensor (3x3x3 entries) :\n %s \n" % result

    #~ first_operation = tf.add(matrix, matrix)
    #~ second_operation = matrix + matrix
    #~ result = session.run(first_operation)
    #~ print "Defined using tensorflow function :"
    #~ print(result)
    #~ result = session.run(second_operation)
    #~ print "Defined using normal expressions :"
    #~ print(result)
    
    third_operation = tf.matmul(matrix, matrix)
    result = session.run(third_operation)
    print("Matrix multiplication with tensorflow")
    print(result)
    
    state = tf.Variable(0)
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)
    # Must init all variables
    init_op = tf.global_variables_initializer()
    
    session.run(init_op)
    print(session.run(state))
    for _ in range(3):
        session.run(update)
        print(session.run(state))
    
    # Placeholder
    a = tf.placeholder(tf.float64)
    b = a * 2
    dictionary={a:[ [ [1,2,3],[4,5,6],[7,8,9],[10,11,12] ] , [ [13,14,15],[16,17,18],[19,20,21],[22,23,24] ] ] }
    result = session.run(b, feed_dict=dictionary)
    print(result)
    
    
