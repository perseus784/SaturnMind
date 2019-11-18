from config import *
import pyqtgraph.opengl as opengl
from pyqtgraph import glColor
from pyqtgraph.Qt import QtCore,QtGui
import sys
from cell_operations import Cell
import numpy as np
import time
import random
from deep_q_network import DeepQnetwork
from collections import deque
class GuiGrid:
    def __init__(self):
        self.app=QtGui.QApplication(sys.argv)
        self.window = opengl.GLViewWidget()
        self.window.setGeometry(0,410,800,800)
        self.window.setCameraPosition(distance=12,azimuth=270)
        x_axis=opengl.GLGridItem()
        x_axis.setSize(x=10,y=10)
        #y_axis=opengl.GLGridItem()
        #y_axis.rotate(90,0,1,0)
        #self.window.addItem(y_axis)
        self.grid=Cell()
        self.dq=DeepQnetwork()
        self.window.addItem(x_axis)
        self.current_node=self.grid.grid_nodes[0]
        self.nodes=opengl.GLScatterPlotItem(pos=self.grid.grid_nodes,color=glColor((0,255,0)),size=7)
        self.goal=opengl.GLScatterPlotItem(pos=self.grid.goal_node,color=glColor((0,0,255)),size=15)
        self.current_node_item=opengl.GLScatterPlotItem(pos=self.current_node,color=glColor((255,0,0)),size=9)
        self.blocked=opengl.GLScatterPlotItem(pos=self.grid.blocked_nodes,color=glColor((255,255,255)),size=13)
        self.counter=0
        self.current_state_queue=deque(maxlen=node_history_size)
        self.next_state_queue=deque(maxlen=node_history_size)
        [self.current_state_queue.append(self.grid.grid_nodes[0]) for i in range(node_history_size)]
        self.generation_counter=0
        self.step_counter=0
        self.tracker=[]
        self.window.addItem(self.nodes)
        self.window.addItem(self.blocked)
        self.window.addItem(self.current_node_item)
        self.window.addItem(self.goal)
        #self.window.show()

    def update(self):
        self.indexed_current_state=[]
        for cr in self.current_state_queue:
            for i,k in self.grid.indexing.items():
                if all(k==cr):
                    self.indexed_current_state.append(i)
                    break
        #get the best next action from q table
        action=self.dq.get_action(self.indexed_current_state)
        #get all the possible you can go from that point
        adjacent_nodes=self.grid.get_adjacent_nodes(self.current_node)
        #getting the next state using the best move that was made
        [[next_node,reward,ac],reached_goal]=self.grid.get_next_node(action,adjacent_nodes)
        self.step_counter+=1

        if reached_goal:
            self.tracker.append([self.generation_counter,self.step_counter])
            print("-------------------------episode over-------------------------------")
            print("generation",self.generation_counter,"number of steps took",self.step_counter)
            #time.sleep(1)
            self.generation_counter+=1
            [self.current_state_queue.append(self.grid.grid_nodes[0]) for i in range(node_history_size)]
            self.indexed_current_state=[]
            for cr in self.current_state_queue:
                for i,k in self.grid.indexing.items():
                    if all(k==cr):
                        self.indexed_current_state.append(i)
                        break
            next_node=self.grid.grid_nodes[0]
            self.step_counter=0

            if generations//2 >= self.generation_counter >=1:
                self.dq.epsilon -= self.dq.decay

            if self.generation_counter%100==0:
                with open('track_file.txt','w') as track_file:
                    track_file.write(str(self.tracker))            
            
        self.indexed_next_state=deque(self.indexed_current_state[:],maxlen=node_history_size)
        for i,k in self.grid.indexing.items():
            if all(k==next_node):
                self.indexed_next_state.append(i)
                break    
        current_q_list=self.dq.predict(self.indexed_current_state)
        next_q_list=self.dq.predict(self.indexed_next_state)
        current_q_list=self.dq.update_q_value(reward,current_q_list[0],next_q_list[0],action)
        _current_one_hot=np.zeros([4])
        _current_one_hot[action]=1
        self.dq.train(self.indexed_current_state,_current_one_hot)
    
        '''
        new_nodes=self.grid.set_rewards(new_nodes)
        print(new_nodes)
        self.current_node,reward,action=random.choice(new_nodes)'''
        self.current_node_item.setData(pos=np.array(next_node))
        self.current_node=next_node
        self.counter+=1
        self.current_state_queue.append(self.current_node)
        print("generation",self.generation_counter,"steps",self.step_counter)
    
    def start(self):
            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                QtGui.QApplication.instance().exec_()

    def animation(self,frametime=10):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(frametime)
        self.start()
    
if __name__ == "__main__":
    g=GuiGrid()
    g.animation()
    
