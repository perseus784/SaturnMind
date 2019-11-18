import tensorflow as tf
import numpy as np
from config import *
import random

class DeepQnetwork:
    def __init__(self):
        self.sess=tf.Session()
        self.one_hot_op=np.eye(no_of_actions)
        self.epsilon=0.7
        self.decay=self.epsilon/((generations//2)-1)
        self.input_placeholder=tf.placeholder(shape=[1,node_history_size],dtype=tf.float32)
        self.label_placeholder=tf.placeholder(shape=[1,no_of_actions],dtype=tf.float32)
        self.network=self.create_network(self.input_placeholder)
        self.loss = tf.reduce_mean(tf.square(self.network - self.label_placeholder))
        self.optimizer=tf.train.AdamOptimizer().minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())  

    '''def get_data(self):
        random_choice=random.randrange(len(input_val))
        return np.array([input_val[random_choice]]),np.array([one_hot_op[random_choice]])'''

    def update_q_value(self,reward,current_q_list,next_q_list,action):
        next_max_q=np.max(next_q_list)
        new_q=reward+discount_factor*next_max_q
        current_q_list[action]=new_q
        return current_q_list

    def create_network(self,input_placeholder):
        network=tf.layers.dense(input_placeholder,units=5,activation=tf.nn.relu)
        network=tf.layers.dense(network,units=5,activation=tf.nn.relu)
        network=tf.layers.dense(network,units=no_of_actions)
        return tf.nn.softmax(network)

    def get_action(self,states):
        if np.random.random()>self.epsilon:
            _action=self.predict(states)
            action=np.argmax(_action)
        else:
            action=np.random.randint(0,no_of_actions)
        return action

    def predict(self,states):
        prediction=self.sess.run(self.network,feed_dict={self.input_placeholder:np.array([states])})
        return prediction

    def train(self,states,output_labels):
        loss,_=self.sess.run([self.loss,self.optimizer],feed_dict={self.input_placeholder:np.array([states]),self.label_placeholder:np.array([output_labels])})
        print("loss",loss)

    '''
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
    '''






