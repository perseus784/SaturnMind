import os
grid_size=4
no_of_blocks=5
no_of_actions=4
no_of_states=((grid_size*2)+1)**2
state_combinations=1
learning_rate=0.001
discount_factor=0.95
generations=10000
epochs=50000
batch_size=32
node_history_size=15000
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
