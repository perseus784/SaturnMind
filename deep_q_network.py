import tensorflow as tf
import numpy as np
from config import *
import random
input_val=np.array(([1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8]),float)
output_val=np.array([[4,5,6,7,8,9]],float)
one_hot_op=np.eye(no_of_actions)
sess=tf.Session()
in_shape=input_val.shape
op_shape=output_val.shape

def get_data():
    random_choice=random.randrange(len(input_val))
    return np.array([input_val[random_choice]]),np.array([one_hot_op[random_choice]])

def create_network(input_placeholder):
    network=tf.layers.dense(input_placeholder,units=5,activation=tf.nn.relu)
    network=tf.layers.dense(network,units=5,activation=tf.nn.relu)
    network=tf.layers.dense(network,units=no_of_actions)
    return tf.nn.softmax(network)

input_placeholder=tf.placeholder(shape=[1,3],dtype=tf.float32)
label_placeholder=tf.placeholder(shape=[1,no_of_actions],dtype=tf.float32)
network=create_network(input_placeholder)
loss = tf.reduce_mean(tf.square(network - label_placeholder))
optimizer=tf.train.AdamOptimizer(0.001).minimize(loss)
sess.run(tf.global_variables_initializer())
for i in range(epochs):
    for batch in range(in_shape[0]//batch_size):
        network_in,actual_output=get_data()
        sess.run([optimizer],feed_dict={input_placeholder:network_in,label_placeholder:actual_output})
    network_in,actual_output=get_data()
    _loss,prediction=sess.run([loss,network],feed_dict={input_placeholder:network_in,label_placeholder:actual_output})
    print(_loss,prediction,actual_output)
        






