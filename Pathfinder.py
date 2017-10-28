import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.ion()

#grid formation
grid_size=5
grid=[[j,i] for j in range(grid_size) for i in range(grid_size)]
print (grid)

#Intialize basic parameters
n_points=grid_size**2
moves=['up','down','left','right']
learning_rate=0.1
episodes=10 #should be adjusted in accordance to grid size
max_reward=0.9
#blue dots
Pitfalls=[5,7,13,19]

#scatterplot format=[[]]
def scatterplot(x,color):
    [plt.scatter(k[0], k[1], c=color,s=20) for k in x]
    pass

#visualize the change in live
def update_gui_grid(k):
    plt.scatter(k[0], k[1], c='g',marker='*',s=80)
    #plt.pause(0.01)
    pass

#gives possible options for any supplied point in the grid
def choices(current_pt,walls):

    #a step in all directions
    available_choices={'right':[current_pt[0]+1,current_pt[1]],
                       'up':[current_pt[0],current_pt[1]+1],
                       'left':[current_pt[0]-1,current_pt[1]],
                       'down':[current_pt[0],current_pt[1]-1]}
    possible_choices=[]

    #remove options which does not come under grid or walls
    [possible_choices.append(a) if (available_choices[a] in grid) and (available_choices[a] not in walls) else [] for a in available_choices]
    return possible_choices

#make a decision based on q_table to proceed to next step
def make_decision(current_point,q_table):
    point=q_table.iloc[current_point]
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
            new_pt=grid.index(pt)
        elif move=='right':
            pt[0] += 1
            new_pt = grid.index(pt)
        elif move=='left':
            pt[0] -= 1
            new_pt = grid.index(pt)
        elif move=='down':
            pt[1] -= 1
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
            scatterplot(grid,'r')#COMMENT TO RUN FASTER
            scatterplot([target], 'g')
            scatterplot(pit_list,'b')
            
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
            update_gui_grid(grid[point])#COMMENT TO RUN FASTER
            plt.clf()#COMMENT TO RUN FASTER

        print('steps.....',steps)

        #plt.close()
    return qtable

if __name__=='__main__':
    
    # print obtained QTABLE
    print(Learner([0, 0], [4, 4]))
    
    plt.show(block=True)
