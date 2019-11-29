import tensorflow as tf
import numpy as np
from config import *
import random
import os

class DeepQnetwork:
    def __init__(self,training_mode):
        self.sess=tf.Session()
        self.epsilon=0.7
        self.decay=self.epsilon/((generations//2)-1)

        if not training_mode:
            saver = tf.train.import_meta_graph(os.path.join(*[ROOT_DIR,"checkpoints","rl_weights-1000.meta"]))
            saver.restore(self.sess,tf.train.latest_checkpoint(os.path.join(*[ROOT_DIR,'checkpoints'])))
            graph = tf.get_default_graph()
            self.predict_network = graph.get_tensor_by_name("dense_4/BiasAdd:0")
            self.predict_placeholder=graph.get_tensor_by_name("Placeholder:0")
        else:
            self.input_placeholder=tf.placeholder(shape=[None,1],dtype=tf.float32)
            self.label_placeholder=tf.placeholder(shape=[None,no_of_actions],dtype=tf.float32)
            self.network=self.create_network(self.input_placeholder)
            self.loss = tf.reduce_mean(tf.square(self.network - self.label_placeholder))
            self.optimizer=tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            '''tf.summary.scalar("loss", self.loss)
            self.writer = tf.summary.FileWriter("tf_summary", graph=tf.get_default_graph())
            self.merged = tf.summary.merge_all()'''
            self.sess.run(tf.global_variables_initializer())  
            #self.saver = tf.train.Saver(max_to_keep=4)

        self.counter=0

    '''def get_data(self):
        random_choice=random.randrange(len(input_val))
        return np.array([input_val[random_choice]]),np.array([one_hot_op[random_choice]])'''
    
    def record_summary(self,_loss,counter):
        summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=_loss)])
        self.writer.add_summary(summary,counter)

    '''def save_model(self):
        self.saver.save(self.sess, os.path.join(*[".","checkpoints","rl_weights"]),global_step=1000)
    
    def load_model(self):
        pass'''

    def update_q_value(self,rewards,current_q_list,next_q_list,actions):
        next_max_qs=np.max(next_q_list,axis=1)
        new_qs=rewards+discount_factor*next_max_qs
        for i in range(len(current_q_list)):
            current_q_list[i,actions[i]]=new_qs[i]
        return current_q_list

    def create_network(self,input_placeholder):
        network=tf.layers.dense(input_placeholder,units=3,activation=tf.nn.relu)
        network=tf.layers.dense(network,units=4,activation=tf.nn.relu)
        network=tf.layers.dense(network,units=5,activation=tf.nn.relu)
        network=tf.layers.dense(network,units=no_of_actions)
        return network

    def get_action(self,state,training_mode):
        if np.random.random()>self.epsilon:
            if training_mode:
                _action=self.predict([state])
            else:
                _action=self.just_predict([state])

            action=np.argmax(_action)
        else:
            action=np.random.randint(0,no_of_actions)
        return action

    def just_predict(self,state):
        prediction=self.sess.run(self.predict_network,feed_dict={self.predict_placeholder:np.array(state)})
        return prediction

    def predict(self,state):
        prediction=self.sess.run(self.network,feed_dict={self.input_placeholder:np.array(state)})
        return prediction

    def train(self,previous_memories):
        self.counter+=1
        previous_memories=np.array(previous_memories).T
        current_nodes,actions,next_nodes,rewards=previous_memories
        current_action_qs=self.predict(np.array([current_nodes]).T)
        next_action_qs=self.predict(np.array([next_nodes]).T)
        current_action_qs=self.update_q_value(rewards,current_action_qs,next_action_qs,actions)
        loss,_=self.sess.run([self.loss,self.optimizer],feed_dict={self.input_placeholder:np.array([current_nodes]).T,self.label_placeholder:current_action_qs})
        print("loss",loss)

        '''if self.counter%500==0:
            self.saver.save(self.sess, os.path.join(*[".","checkpoints","rl_weights"]),global_step=1000)'''

        #self.record_summary(loss,self.counter)





