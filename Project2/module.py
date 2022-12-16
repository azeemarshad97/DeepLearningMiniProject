import torch
import math

torch.set_grad_enabled(False)

class Layer(object):

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self.weights = torch.randn(n_inputs, n_outputs)
        # How to handle bias?
        # self.bias = torch.randn(n_outputs)
        self.gradient = torch.zeros(n_inputs, n_outputs)

    def forward(self , x: torch.Tensor) -> torch.Tensor:
        # do forward pass with input x and weight weights
        raise NotImplementedError


    def backward(self , *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def __call__(self, x):
        return self.forward(x)


class Linear(Layer):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.gradient = x
        return x @ self.weights

    def backward(self, *gradwrtoutput):
        return self.weights.T @ gradwrtoutput @ self.gradient

    def param(self):
        return [self.weights, self.gradient]


class ActivationLayer(Layer):
    
    @staticmethod
    def relu(self, x: torch.Tensor):
        return x.max(torch.zeroslike(x))

    @staticmethod
    def tanh(self, x: torch.Tensor):
        pass

    def __init__(self, method: str):
        methods = {
            'tanh': self.tanh,
            'relu': self.relu
        }
        try:
            self.method = methods[method]
        except KeyError:
            raise ValueError('Wrong method specified, got ', method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.method(x)

    def backward(self, *gradwroutput):
        return gradwroutput
   
    

class Loss(Object):

    @staticmethod
    def mse(x, y):
        pass

    def __init__(self, loss: str):
        losses = {
                'mse': self.mse,
            }
        try:
            self.loss = losses[loss]
        except KeyError:
            raise ValueError('Wrong loss specified, got ', loss)



class Sequential(Object):
    def __init__(self, layers, loss: str):
        pass

    def forward(self, x):
        for layer in self.layers:
            x = layer.(x)

if __name__ == '__main__':
    model = Sequential() 