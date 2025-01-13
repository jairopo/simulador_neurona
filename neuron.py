import numpy as np

class Neuron:
    def __init__(self, weights, bias, func):
        self.w = weights
        self.b = bias
        self.f = func
    
    def changeBias(self, bias):
        self.b = bias
    
    def run(self, input_data):
        self.__x = input_data
        # Check both lengths
        if len(self.__x) == len(self.w):
            # Get the y value
            y = sum([self.__x[i] * self.w[i] for i in range(len(self.__x))]) + self.b
            
            # Check the function to run
            if self.f == "ReLu":
                return f"ReLu: {max(0, y)}"
            elif self.f == "Tangente hiperbólica":
                return f"Tanh: {np.tanh(y)}"
            elif self.f == "Sigmoide":
                return f"Sigmoid: {1 / (1 + np.exp(-y))}"
                
        