import numpy as np
input_layer=3
hidden_layer = 5
output_layer =1
l_rate = 0.3
weight1=np.random.rand(input_layer,hidden_layer)
weight2=np.random.rand(hidden_layer,output_layer)

input_val=np.array(([1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8]),float)
input_val=input_val/np.amax(input_val,0)
output_val=np.array([[4,5,6,7,8,9]],float).T
output_val=output_val/np.amax(output_val,0)

def activation_function(y):
    return 1/(1+np.exp(-y))

def backprop_derivative(y):
    return activation_function(y)*(1-activation_function(y))

def neuron_operation(layer1,layer2):
    result=np.dot(layer1,layer2)
    return activation_function(result)

def get_network(input_placeholder):
    network=[]
    network.append(neuron_operation(input_placeholder,weight1))
    network.append(neuron_operation(network[0],weight2))
    return network

def do_backprop(network,actual_output):
    global weight1
    global weight2
    error=network[-1] - actual_output
    temp=np.dot(error*backprop_derivative(actual_output),weight2.T)
    print(temp.shape)
    temp2=temp*error*backprop_derivative(network[0])
    print(temp2.shape)
    weight1 += l_rate * np.dot(input_val.T,temp2)
    weight2 += l_rate * np.dot(network[0].T,error*backprop_derivative(actual_output))
    print(weight1,weight2)
    pass

for i in range(3):
    network=get_network(input_val)
    print(network[-1],output_val)
    do_backprop(get_network(input_val),output_val)

