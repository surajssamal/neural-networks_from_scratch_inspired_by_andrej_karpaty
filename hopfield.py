#!/bin/python3
import numpy as np 

class Hopfield:
    def __init__(self,num_of_neuron):
        self.neuron = num_of_neuron
        self.weights = np.zeros((self.neuron,self.neuron))
    def train(self,data):
        for i in data:
            self.weights += np.outer(i,i)
        np.fill_diagonal(self.weights,0)
        self.weights /= self.neuron

            
    def recall(self,data_test,steps=50):
        recalled_pattern = data_test.copy()
        for _ in range(steps):
            for i in range(self.neuron):
                weighted_sum = np.dot(self.weights[i],recalled_pattern)
                recalled_pattern[i] = 1 if weighted_sum >=0 else -1
        return recalled_pattern
if __name__ == "__main__":
    
    pattern = np.array([[1, -1, 1, -1, 1, -1, 1, -1],[-1, -1, 1, 1, -1, -1, 1, 1]])
    random_tnsor = np.array([1, -1, -1, -1, 1, -1, 1, -1])
    model = Hopfield(8)
    model.train(pattern)
    print(model.recall(random_tnsor))


