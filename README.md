# SaturnMind
A repo to understand Reinforcement Learning, the first step towards general AI.


## Introduction :
**Step 1** : Define your environment and set your actions, goal.  
            
            E.g.
            Environment : Super Mario.(The Best game ever!)    
            Actions : move forward, jump, duck, long jump etc. 
            Goal : Retrieving the princess.
<p align="center">
<img src="/media/environment.gif" alt="Super Mario Environment" width="550" height="300">
</p>  

*Note: enemy tortoises and triangle shaped things are removed from the scenerio for simplicity.*  


**Step 2** : Initialize Q table wth states and actions.  

            States- Current state or Position in the environment.
                    E.g. Current location of Mario in the frame.
            Action- List freedom of movements in the environment that is defined.  
  
  Like this one :  
  
 | State | Forward | Jump | Duck | Ljump |
 |-------|---------|------|------|-------|
 | Frame 0     | 0.0     | 0.0  | 0.0  | 0.0   |
 | Frame 1     | 0.0     | 0.0  | 0.0  | 0.0   |
 | Frame 2     | 0.0     | 0.0  | 0.0  | 0.0   |

***Now our job is to train and adapt the above Q table by interacting with the environment in following steps*** 

**Step 3** : Let the hero explore environment.

            Our hero can take a random move if Q table's Move is zero or equally distributed. 
            Else hero has to choose the move with highest reward for the present state.
            
            For a given State:
                 if Jump > Forward:
                       Mario chooses to Jump.
                 else:
                       Mario chooses Forward.
 <p align="center">
<img src="/media/Exploring_env.gif" alt="Super Mario" width="550" height="300">
</p>
 
 
 **Step 4** : Update the Q table.
    
          Now Rewards for each move towards the goal is calculated and updated in Q table.
          It is specific to that State and Move at that instant.
             
          Q table( State , move) = Q table( State , move) + learning_Rate *[Q table(current(S,M) - previous(S,M))]
          
          learning_rate= 0.1 # one step at a time.
          S - State , M - Move.
             
             
A updated Q table after some movements:

| State   | Forward | Jump | Duck | Ljump |
|---------|---------|------|------|-------|
| Frame 0 | 0.4     | 0.0  | 0.2  | 0.9   |
| Frame 1 | 0.6     | 0.3  | 0.0  | 0.5   |
| Frame 2 | 0.9     | 0.7  | 0.0  | 0.0   |

**Step 5** : Handling Fail conditions.

            If our hero fails to reach the goal, Update Q table with a negative reward.
            Negative rewarding a Move at that State reduces the selection of that movement in future.
<p align="center">
<img src="/media/Fail.gif" alt="Super Mario" width="550" height="300">
</p>

**Step 6** : Reaching the goal.
            
            The above process is continued till our hero reaches the goal.
            Once the Goal is reached, Our program completed a generation.  
            
<p align="center">
<img src="/media/Goal.gif" alt="Super Mario Environment" width="550" height="300">
</p>  

**Step 7** : Passing Knowledge to Generations.

            Once a generation is complete, game is started again.
            But the same Q table is kept, inorder to have knowledge of the previous generations.
            The Steps 3 - 6 is repeated again and again till Saturation or till enough experience in large cases.
 
 ## Improvise. Adapt. Overcome.
 
 *Finally, we got our updated Q table with enough knowledge of the environment.
This Q table can be used to successfuly complete Super Mario with much ease.*  

<p align="center">
<img src="/media/savage.gif" alt="Mario Purge" width="550" height="300">
</p>

*Nostalgia, huh?*

Down below are some programs written in Python to demonstrates the above steps lively. 
___
            
       


## Projects :
1. [**Grid Pathfinder**](https://github.com/perseus784/SaturnMind/blob/master/Pathfinder.py) & 
[**Grid Pathfinder 3D**](https://github.com/perseus784/SaturnMind/blob/master/Pathfinder_3D.py) :

*Prerequisite :*
* Pandas
* Numpy
* Matplotlib

A program to navigate through a grid even with blocks in the grid.
Code segments are explained in commented lines.
Qlearning table is used.
Feel free to experiment with the variables.
<p align="center">
<img src="/media/Pathfinder_3D.gif" alt="Mario Purge" width="600" height="350">
</p>

!!
