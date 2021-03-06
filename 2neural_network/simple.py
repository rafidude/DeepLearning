import numpy as np

def sigmoid(x):
    # TODO: Implement sigmoid function
    return (1.0/(1.0 + np.exp(-x)))

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
print("Inputs: ", inputs)
print("Weights: ", weights)
bias = -0.1

# TODO: Calculate the output
output = sigmoid(np.dot(inputs, weights) + bias)

print('Output:')
print(output)