import numpy as np
from config import *

class Qlearning:

    def __init__(self,no_of_actions,no_of_states,state_combinations):
        self.q_table=np.random.uniform(low=-1,high=1,size=[state_combinations,no_of_states,no_of_actions])
    
    def update_qtable(self,reward,state,action,next_state):
        next_max_q=np.max(self.q_table[:,next_state,:])
        current_q=self.q_table[:,state,action]
        new_q=(1-learning_rate)*current_q+learning_rate*(reward+discount_factor*next_max_q)
        self.q_table[:,state,action]=new_q
    
    def get_action(self,state):
        action=np.argmax(self.q_table[:,state,:])
        return action


