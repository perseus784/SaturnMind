import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import os, subprocess
from pylab import *

import tensorflow as tf
import pandas as pd
import time
grid_size=5
n_points=5**3
grid=[[k,j,i] for k in range(grid_size) for j in range(grid_size) for i in range(grid_size)]
print(grid)
moves=['up','down','left','right','fwd','back']
learning_rate=0.1
episodes=150 #should be adjusted in accordance to grid size
max_reward=0.9
plt.ion()
#blue dots
Pitfalls=[5,13,24,75,92,61,120,101]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#scatterplot format=[[]]
def scatterplot(x,color):
    [ax.scatter(k[0], k[1], k[2], c=color,s=20) for k in x]
    pass

#visualize the change in live
def update_gui_grid(k):
    ax.scatter(k[0], k[1], k[2],c='g',marker='*',s=80)
    pass

#gives possible options for any supplied point in the grid
def choices(current_pt,walls):

    #a step in all directions
    available_choices={'right':[current_pt[0]+1,current_pt[1],current_pt[2]],
                       'up':[current_pt[0],current_pt[1]+1,current_pt[2]],
                       'left':[current_pt[0]-1,current_pt[1],current_pt[2]],
                       'down':[current_pt[0],current_pt[1]-1,current_pt[2]],
                       'fwd':[current_pt[0],current_pt[1],current_pt[2]+1],
                       'back':[current_pt[0],current_pt[1],current_pt[2]-1]}
    possible_choices=[]

    #remove options which does not come under grid or walls
    [possible_choices.append(a) if (available_choices[a] in grid) and (available_choices[a] not in walls) else [] for a in available_choices]
    return possible_choices

#make a decision based on q_table to proceed to next step
def make_decision(current_point,q_table):
    point = q_table.iloc[current_point]
    #constrain the moves within grid and wall
    moves=choices(grid[current_point],[[]])#add walls here
    move=''
    # if no rewards on any sides or equal rewards, then take a random move.
    if (np.random.uniform()>0.9) or (point.all()==0):
        move=np.random.choice(moves)
    else:#choose the one with maximum reward
        move=point.argmax()
    return move

#making change in the grid environment
def updating_environment(point,move,destination):
    pt=grid[point][:]
    new_pt=0
    #if destination do nothing
    if point==destination:
        new_pt=destination
    else:#return the new point
        if move=='up':
            pt[1] +=1
        elif move=='right':
            pt[0] += 1
        elif move=='left':
            pt[0] -= 1
        elif move=='down':
            pt[1] -= 1

        elif move =='fwd':
            pt[2] += 1
        elif move =='back':
            pt[2] -= 1
        new_pt = grid.index(pt)
    return new_pt

def Learner(start,target):

    #display grid
    scatterplot(grid,'r')
    plt.pause(0.01)
    scatterplot([start],'b')
    plt.pause(0.01)
    scatterplot([target],'g')
    plt.pause(0.01)

    # Initialize q-table
    qtable=pd.DataFrame(np.zeros((n_points,len(moves))),columns=moves)

    update_gui_grid(start)
    pit_list=[grid[i] for i in Pitfalls]

    for epi in range(episodes):

        #getting point indices
        point = grid.index(start)
        destination = grid.index(target)
        print('Generation......',epi)
        reached=False
        steps=0

        while not reached:
            '''
            scatterplot(grid,'r')#COMMENT TO RUN FASTER
            scatterplot([target], 'g')
            scatterplot(pit_list,'b')'''

            #get the next move that should be made
            move=make_decision(point,qtable)

            #get reaction of the environment to the selected move
            next_pt=updating_environment(point,move,destination)

            #take q_value for the next value for the above move
            q_next=qtable.ix[next_pt,move]

            #if reached destination give maximum reward and end loop
            if point==destination:
                q_target=1
                reached=True

            #Negative rewards for pitfalls
            elif point in Pitfalls:
                q_target=-1

            #else continue to search for target
            else:

                #selecting the value which has maximum rewards in that point
                q_target=max_reward*qtable.iloc[next_pt].max()
                #choosing the max inorder to keep this reward highest for the point

            #IMP: updating the q_value for that movement
            qtable.ix[point,move] += learning_rate*(q_target-q_next)

            #updating values
            point=next_pt
            steps +=1

            if epi==149:
                update_gui_grid(grid[point])
                draw()  # COMMENT TO RUN FASTER
                savefig('mat/picture' + str(steps))


        print('steps.....',steps)

        #plt.close()

    return qtable



if __name__=='__main__':
    # print obtained QTABLE
    print(Learner([0, 0,0], [4, 4,4]))

    plt.show(block=True)
