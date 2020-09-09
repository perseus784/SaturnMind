# SaturnMind
Custom implementation of Q learning and Deep Q learning.
Reinforcement Learning Explained here: https://medium.com/@ipaar3/saturnmind-94586f0d0158

## Requirements:
* PyQt5, OpenGL
* TensorFlow
* Numpy
 
## Environment:
<p align="center">	
<img src="/media/exploration.gif" alt="Grid Environment" width="850" height="450">	
</p>	

# Q learning:

Run [grid_environment.py](https://github.com/perseus784/SaturnMind/blob/master/Qlearning/grid_environment.py) to run just the Qlearning.
The experimental setup is simple, it's a grid with 8x8 nodes and some blocks. The aim is to navigate to the diagonally opposite side of the grid with minimal number of steps.
<p align="left">	
<img src="/media/qlearningcli.gif" alt="qlearning after training" width="400" height="400">	
<img src="/media/qlearninggui.gif" alt="qlearning after training" width="400" height="400">	
</p>  
<p align="center">
<img src="https://github.com/perseus784/SaturnMind/blob/master/media/qlearning_graph.png" width="800" height="400">
</p>
Since there is a lot of blogs on Qlearning and how it works, lets move to Deep Q learning.

# Deep Q learning:
As the state space grows and number of actions gets complex, we cannot use the Q table techinque. It will cause a memory overflow or even if you have huge memory, it is just not efficient. That's where neural networks come in play. These neural networks can memorize the environment with very less memory and sometimes even find a generalized way of solving the problem.  

For example, if we go for very complex games like Call of Duty, there is a lot of states and requires complex set of actions to reach the next state.  Say killing an opponent requires a series of actions with multiple weapons. while Q table method will record the set of actions when the user has to kill the next opponent it's again a new state for it. it has to record it as a new state again since it doesn't know to generalize. But in case of a neural network, it learns how to kill a person and it records it as a new state. So, it can just generalize the kill action throughout the environment. That's exactly why neural networks are better than Q table. But still Q table method will require less time to converge in a smaller state space with less set of actions. 

## About Network:
Finding the hyperparameters is done through lot of experimentation and unit testing. Please tell me if you find a better optimized network for the task. 
> You might have to use a CNN if your state space defined by images. [Simple Tutorial for CNN using Tensorflow.](https://github.com/perseus784/BvS)
Batch Size - 32

## Architecture: 
<p align="center">
<img src="https://github.com/perseus784/SaturnMind/blob/master/media/deepQlearning.png" width="800" height="400">
</p>

### Training:
    python grid_environment.py --train True

Convergence is observed after 4000 - 5000 episodes 

<p align="left">	
<img src="/media/trained_cli.gif" alt="qlearning after training" width="400" height="400">	
<img src="/media/trained.gif" alt="qlearning after training" width="400" height="400">	
</p> 

<p align="left">	
<img src="/media/deep_q_learning_graph.png" alt="qlearning after training" width="400" height="400">	
<img src="/media/tensorflow_loss.png" alt="qlearning after training" width="400" height="400">	
</p> 

### Saving the model:
See [this](https://github.com/perseus784/BvS#saving-ourmodel) to know how to save and retrieve the model.

### Prediction:
After the model is trained for 5000 episodes, we can just quit the environment and the model is saved automatically.

    python grid_environment.py 

## What's really happening?
For each prediction, we'll get q values for the 4 actions. 

            cost = predicted q values - actual q values.
            but the actual values here depends on the bellman's equation.
            actual qvalue= reward + discount_factor * max qvalue 
            max qvalue is from the previous state taken to go to the next state.
            
            now optimize the cost using Adam Optimizer.
            
            
# Reinforcement Learning Explained:

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


**Step 2** : Initialize Q table wth states and actions.  	Network Parameters:
Q table AKA Quality table represents the quality of move that is being made on that state.  	
*Higher Magnitude -> Higher Quality Move* in a state.	


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
             	
             	
An updated Q table after some movements:	

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

