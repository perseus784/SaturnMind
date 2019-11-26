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
import argparse

class GuiGrid:
    def __init__(self,training_mode):
        self.training_mode=training_mode
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
        self.dq=DeepQnetwork(training_mode)
        self.window.addItem(x_axis)
        self.current_node=self.grid.grid_nodes[0]
        self.nodes=opengl.GLScatterPlotItem(pos=self.grid.grid_nodes,color=glColor((0,255,0)),size=7)
        self.goal=opengl.GLScatterPlotItem(pos=self.grid.goal_node,color=glColor((0,0,255)),size=15)
        self.current_node_item=opengl.GLScatterPlotItem(pos=self.current_node,color=glColor((255,0,0)),size=9)
        self.blocked=opengl.GLScatterPlotItem(pos=self.grid.blocked_nodes,color=glColor((255,255,255)),size=13)
        self.counter=0
        self.previous_memory=deque(maxlen=node_history_size)
        self.generation_counter=0
        self.step_counter=0
        self.tracker=[]
        self.window.addItem(self.nodes)
        self.window.addItem(self.blocked)
        self.window.addItem(self.current_node_item)
        self.window.addItem(self.goal)
        self.window.show()

    def get_batch(self,sampling_size):
        return random.sample(self.previous_memory,sampling_size)

    def update(self):
        for i,k in self.grid.indexing.items():
            if all(k==self.current_node):
                indexed_current_node=i

        #get the best next action from q table
        action=self.dq.get_action([indexed_current_node],self.training_mode)

        #get all the possible you can go from that point
        adjacent_nodes=self.grid.get_adjacent_nodes(self.current_node)

        #getting the next state using the best move that was made
        [[next_node,reward,ac],reached_goal]=self.grid.get_next_node(action,adjacent_nodes)

        self.step_counter+=1
        for i,k in self.grid.indexing.items():
            if all(k==next_node):
                indexed_next_node=i

        if self.step_counter>3000:
            return

        self.previous_memory.append([indexed_current_node,action,indexed_next_node,reward])

        if reached_goal:
            self.tracker.append([self.generation_counter,self.step_counter])
            print("-------------------------episode over-------------------------------")

            print("generation",self.generation_counter,"number of steps took",self.step_counter)

            self.generation_counter+=1

            next_node=self.grid.grid_nodes[0]
            self.step_counter=0

            if generations//2 >= self.generation_counter >=1:
                self.dq.epsilon -= self.dq.decay
            
            if self.generation_counter%10==0:
                with open('track_file.txt','w') as track_file:
                    track_file.write(str(self.tracker))   
        
        if self.training_mode:
            if len(self.previous_memory)>batch_size:
                previous_memories=self.get_batch(batch_size)
                self.dq.train(previous_memories)
    
        self.current_node_item.setData(pos=np.array(next_node))
        
        self.current_node=next_node
        self.counter+=1
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
    parser=argparse.ArgumentParser(description='dqrl')
    parser.add_argument('--train',dest="training_mode",type=bool,default=False)
    args=parser.parse_args()
    g=GuiGrid(args.training_mode)
    g.animation()
    
