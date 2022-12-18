import torch
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np # For plotting

import module as m

torch.set_grad_enabled(False)

N_SAMPLES = 1000

if __name__ == '__main__':

    print('Generating data...')
    # Generate train
    X_train = torch.rand(N_SAMPLES, 2)
    y_train = torch.ones(N_SAMPLES)
    X_test = torch.rand(N_SAMPLES, 2) 
    y_test = torch.ones(N_SAMPLES)

    # Set targets to 0 if outside disk centered at (0.5, 0.5) with radius 1/sqrt(2*pi)

    y_train[
        torch.norm(X_train - torch.tensor([0.5, 0.5]), dim=1) \
        > 1/math.sqrt(2*math.pi)
        ] = 0
    y_test[
        torch.norm(X_test - torch.tensor([0.5, 0.5]), dim=1) \
        > 1/math.sqrt(2*math.pi)
        ] = 0
        

    # plt.figure(figsize=(5,5))
    # plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='bwr', s=8)
    # plt.show()

    print('Training model...')
    model = m.Sequential(
    m.LinearLayer(2, 25),
    m.ReLU(),
    m.LinearLayer(25, 25),
    m.Tanh(),
    m.LinearLayer(25, 25),
    m.Tanh(),
    m.LinearLayer(25, 25),
    m.Tanh(),
    m.LinearLayer(25, 1),
    m.Sigmoid(),
)

    optim = m.SGD(model, lr=0.01)

    epochs = 100
    batch_size = 100
    n_batches = N_SAMPLES // batch_size

    print()
    for i in range(epochs):
        # Get random batch
        idx = torch.randint(0, N_SAMPLES, (batch_size,))
        for j in range(n_batches):
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            loss = optim(X_batch, y_batch.unsqueeze(1), debug=False)

        print(f'Epoch {i+1}/{epochs}, Loss: {loss}', end='\r')

    print()
    print('Testing model...')
    y_pred = model(X_test)
    y_pred = y_pred > 0.5

    acc = torch.sum(y_pred.squeeze() == y_test) / N_SAMPLES
    print(f'Accuracy: {acc}')


    plt.figure(figsize=(5,5))

    mask1 = (y_pred==1).squeeze()
    mask0 = (y_pred==0).squeeze()

    plt.scatter(X_test[:,0][mask1], X_test[:,1][mask1], s=8, label=1)
    plt.scatter(X_test[:,0][mask0], X_test[:,1][mask0], s=8, label=0)


    # plot unit circle
    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(
        0.5 + np.cos(t)/math.sqrt(2*math.pi), 
        0.5 + np.sin(t)/math.sqrt(2*math.pi), 
        c='k', label='unit circle', lw=1, alpha=0.5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()