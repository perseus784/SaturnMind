# SaturnMind
A repo to understand Reinforcement Learning, a first step towards general AI.


## Introduction :
**Step 1** : Define your environment and set your actions, goal.  
            
            E.g.
            Environment : Super Mario.(The Best game ever!)    
            Actions : move forward, jump, duck, long jump etc. 
            Goal : Retrieving the princess.
<p align="center">
<img src="http://9to5animations.com/wp-content/uploads/2016/03/mario-gif-animated.gif" alt="Super Mario Environment" width="550" height="300">
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
<img src="https://media.giphy.com/media/gjdS9VFkMHzva/giphy.gif" alt="Super Mario" width="550" height="300">
</p>
 
 
 **Step 4** : Update the Q table.
    
             Now Rewards for each move towards the goal is calculated and updated in Q table.
             It is specific to that State and Move at that instant.
             
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
<img src="http://www.gifbin.com/bin/25yu3420sw.gif" alt="Super Mario" width="550" height="300">
</p>

**Step 6** : Reaching the goal.
            
            The above process is continued till our hero reaches the goal.
            Once the Goal is reached, Our program completed a generation.  
            
<p align="center">
<img src="http://img38.laughinggif.com/pic/HTTP2Nkbi5pbnF1aXNpdHIuY29tL3dwLWNvbnRlbnQvdXBsb2Fkcy8yMDEzLzA0L0ltZ3VyLU1lZXRzLVNvLU1hcmlvLUJyb3MuZ2lm.gif" alt="Super Mario Environment" width="550" height="300">
</p>  

**Step 7** : Passing Knowledge to Generations.

            Once a generation is complete, game is started again.
            But the same Q table is kept, inorder to have knowledge of the previous generations.
            The Steps 3 - 6 is repeated again and again till Saturation or till enough experience in large cases.
 
 ## Improve. Adapt. Overcome.
 
 *Finally, we got our updated Q table with enough knowledge of the environment.
This Q table can be used to successfuly complete Super Mario with much ease.*  

<p align="center">
<img src="https://i.imgur.com/hwCwZgV.gif" alt="Mario Purge" width="550" height="300">
</p>

*Nostalgia, huh?*

Down below are some projects which demonstrates the above steps in Python lively. 
___
            
       


## Projects :
1. [**Grid Pathfinder**](https://github.com/perseus784/SaturnMind/blob/master/Pathfinder.py) :

*Prerequisite :*
* Pandas
* Numpy
* Matplotlib


A program to navigate through a grid even with blocks in the grid.
Code segments are explained in commented lines.
Qlearning table is used.
Feel free to experiment with the variables.

!!
