import numpy as np
from config import *
import random
random.seed(132)
class Cell:
    def __init__(self):
        counter=0
        self.grid_nodes=np.array([[i,j] for j in range(-grid_size,grid_size+1) for i in range(-grid_size,grid_size+1)])
        #self.grid_nodes=np.array([[j,i] for i,j in enumerate(_nodes)])
        self.goal_node=self.grid_nodes[-1]
        self.blocked_nodes=np.array(random.sample(list(self.grid_nodes),no_of_blocks))
        self.indexing={i:j for i,j in enumerate(self.grid_nodes)}

    def set_rewards(self,nodes):
        rewarded_nodes=[]
        for i in nodes:
            j=i[0]
            if  (self.blocked_nodes==j).all(1).any() or not (self.grid_nodes==j).all(1).any():
                reward=-1
            elif np.array_equal(np.array(j),np.array(self.goal_node)):
                reward=1
            else:
                reward=-1
            rewarded_nodes.append([j,reward,i[-1]])
        return np.array(rewarded_nodes)

    def get_adjacent_nodes(self,current_node):
        x_dec=current_node[0]-1 if current_node[0]>-grid_size else -grid_size+1
        y_dec=current_node[1]-1 if current_node[1]>-grid_size else -grid_size+1
        x_inc=current_node[0]+1 if current_node[0]<grid_size else grid_size
        y_inc=current_node[1]+1 if current_node[1]<grid_size else grid_size

        left=[[x_dec,current_node[1]],"left"]
        right=[[x_inc,current_node[1]],"right"]
        fwd=[[current_node[0],y_inc],"fwd"]
        bwd=[[current_node[0],y_dec],"bwd"]

        #print([left,right,fwd,back])
        return self.set_rewards(np.array([left,right,fwd,bwd]))
    
    def get_next_node(self,action,adjacent_nodes):
        next_node= adjacent_nodes[action]
        if np.array_equal(np.array(next_node[0]),np.array(self.goal_node)):
            reached_goal=True
        else:
            reached_goal=False
        return [next_node,reached_goal]
    






