import torch
import math

torch.set_grad_enabled(False)

class Layer(object):

    name = ''

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = torch.randn(n_inputs, n_outputs)
        self.bias = torch.randn(n_outputs)
        self.gradwrtw = torch.zeros(n_inputs, n_outputs)
        self.gradwrtb = torch.zeros(n_outputs)


    def forward(self , x: torch.Tensor) -> torch.Tensor:
        # do forward pass with input x and weight weights
        raise NotImplementedError


    def backward(self , *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
    
    def __call__(self, x):
        return self.forward(x)


class LinearLayer(Layer):

    name = 'Linear'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x
        self.output = x @ self.weights + self.bias
        return self.output

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        self.gradwrtw = gradwrtoutput.T @ self.input
        self.gradwrtb = gradwrtoutput.mean(0)
        return gradwrtoutput @ self.weights.T

    def param(self):
        return [self.weights, self.bias]


class ReLU(Layer):

    name = 'ReLU'

    def __init__(self):
        super().__init__(0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x
        self.output = x.max(torch.zeros_like(x))
        return self.output

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        return gradwrtoutput * (self.input > 0).float()
   
class Tanh(Layer):

    name = 'Tanh'

    def __init__(self):
        super().__init__(0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x
        self.output = (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
        return self.output

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        return gradwrtoutput * (1 - self.output ** 2)

class Sigmoid(Layer):

    name = 'Sigmoid'

    def __init__(self):
        super().__init__(0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x
        self.output = 1 / (1 + (-x).exp())
        return self.output

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        return gradwrtoutput * self.output * (1 - self.output)
    

class LossMSE(Layer):

    name = 'LossMSE'

    def __init__(self):
        super().__init__(0, 0)
        self.gradient = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.x = x
        self.y = y
        self.backward()
        return ((x - y) ** 2).mean()

    def backward(self) -> torch.Tensor:
        self.gradient = 2 * (self.x - self.y)
        return self.gradient

    def __call__(self, x, y):
        return self.forward(x, y)


class Sequential:

    def __init__(self, *layers: Layer, loss: str= 'MSE'):
        self.layers = []
        for layer in layers:
            self.layers.append(layer)
        
        losses = {
            'MSE': LossMSE()
        }
        try:
            self.loss = losses[loss]
        except KeyError:
            raise ValueError('Loss function not implemented')

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, x: torch.Tensor, debug=False) -> torch.Tensor:
        if debug: 
            print('-' * 20)
            print('Forward pass:')
        for i, layer in enumerate(self.layers):
            if debug: x_ = x
            x = layer(x)
            if debug:
                print(f'Layer {i} {layer.name}: [{list(x_.shape)}] X [{list(layer.weights.shape) if len(layer.param())>0 else None}] -> [{list(x.shape)}]')
        return x
    
    def backward(self, gradwrtoutput=None, debug=False) -> torch.Tensor:
        if debug: 
            print('-' * 20)
            print('Backward pass:')
        if self.loss.gradient is not None:
            gradwrtoutput = self.loss.gradient
        else:
            raise RuntimeError('Loss function not called')
        for i, layer in enumerate(reversed(self.layers)):
            if debug: g_ = gradwrtoutput
            gradwrtoutput = layer.backward(gradwrtoutput)
            if debug: 
                print(f'Layer {i} {layer.name}: [{list(g_.shape)}] X [{list(layer.weights.T.shape) if len(layer.param())>0 else None}] -> [{list(gradwrtoutput.shape)}]')
        return gradwrtoutput

    def param(self):
        params = []
        for layer in self.layers:
            params.append(layer.param())
        return params

    def __call__(self, x, debug=False):
        return self.forward(x, debug=debug)


class Optimizer:

    def __init__(self, model: Sequential, lr: float):
        self.model = model
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for layer in self.model.layers:
            layer.gradwrtw = torch.zeros_like(layer.gradwrtw)

    def __call__(self, x, y, debug=False):
        self.zero_grad()
        y_pred = self.model(x, debug=debug)
        loss = self.model.loss(y_pred, y)
        if debug:
            print('-----')
            print('Loss')
            print('Loss:', loss)
            print('Gradient:', self.model.loss.gradient.shape)
        self.model.backward(debug=debug)
        self.step()
        return loss

class SGD(Optimizer):

    def __init__(self, model: Sequential, lr: float):
        super().__init__(model, lr)

    def step(self):
        for layer in self.model.layers:
            layer.weights -= self.lr * layer.gradwrtw.T
            layer.bias -= self.lr * layer.gradwrtb









# class Adam(Optimizer):
    
#     def __init__(self, model: Sequential, lr: float, betas: tuple=(0.9, 0.999), eps: float=1e-8):
#         super().__init__(model, lr)
#         self.betas = betas
#         self.eps = eps
#         self.m = []
#         self.v = []
#         for layer in self.model.layers:
#             self.m.append(torch.zeros_like(layer.weights))
#             self.v.append(torch.zeros_like(layer.weights))

#     def step(self):
#         for i, layer in enumerate(self.model.layers):
#             self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * layer.gradient
#             self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * layer.gradient ** 2
#             m_hat = self.m[i] / (1 - self.betas[0])
#             v_hat = self.v[i] / (1 - self.betas[1])
#             layer.weights -= self.lr * m_hat / (v_hat.sqrt() + self.eps)