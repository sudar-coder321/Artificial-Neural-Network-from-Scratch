import math
 
import numpy as np
 
def sigmoid(x):
    return 1/(1+np.exp(-x))
 
def sigmoid_derivtive(x):
  return sigmoid(x)* (1-sigmoid(x))
 
 
inputs = [0.05,0.1]  # x1 and x2 => 2 inputs
real_outputs = [0.01,0.99] # y1 and y2 => 2 final outputs
weights_for_hidden = [[0.15,0.2],[0.25,0.3]] # weights w1,w2,w3,w4 => input from input layer to the hidden layer 
weights_for_output = ([[0.4,0.45],[0.5,0.55]]) #weights w5,w6,w7,w8 => input from hidden layer to the output layer
 
hidden_bias = 0.35 # in this case we take a single bias value, we can take multiple bias values
 
output_bias = 0.6
 
h=np.dot(weights_for_hidden,inputs)+hidden_bias
 
h_activate = sigmoid(h)
 
print(h_activate)
 
 
 
obtained_output = np.dot(weights_for_output,h_activate)+output_bias
 
o_activate = sigmoid(obtained_output)
 
 
error_mse = np.square(np.subtract(real_outputs,o_activate)).mean()
 
print(o_activate,error_mse)
 
 
# chain rule diffentiation of the derivative function to be used in backpropagation
 
 
learning_rate = 0.5
 
for i in range(1000):
 
  d1 = o_activate[0]-real_outputs[0] #--> 1st partial deriv target_output - real_output
 
  d2 = o_activate[0]*(1-o_activate[0])  # --> 2nd partial deriv target_output*(1-target_output)
 
  d3 = h_activate[0] # 3rd partial deriv --> input
 
  d4 = o_activate[1]-real_outputs[1]
 
  d5 = o_activate[1]*(1-o_activate[1])
 
  d6 = h_activate[1]
 
  derror_by_dw5 =  derror_by_dw7 = d1* d2 *d3
 
  derror_by_dw6 = derror_by_dw8 = d4* d5* d6
 
  
 
  weights_for_output[0][0] -= learning_rate*derror_by_dw5
  weights_for_output[0][1] -= learning_rate*derror_by_dw6
  weights_for_output[1][0] -= learning_rate*derror_by_dw7
  weights_for_output[1][1] -= learning_rate*derror_by_dw8
 
  obtained_output = np.dot(weights_for_output,h_activate)+output_bias
 
  o_activate = sigmoid(obtained_output)
 
 
  error_mse = np.square(np.subtract(real_outputs,o_activate)).mean()
 
  print(o_activate,error_mse)
