# Source https://github.com/IJCAI2020-SGES/SGES
import numpy as np
from copy import deepcopy
import time
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class GES:
    def __init__(self, loader, model, criterion):
        self.loader = loader
        self.model = model
        self.criterion = criterion
    
    # Guided-ES framework to estimate gradients
    def ges_compute_grads(self, x, loss_fn, U, k=1, pop_size=1, sigma=0.1, alpha=0.5):
        # Globalspace param
        a = sigma * np.sqrt(alpha / x.shape[0])
        # Subspace param
        c = sigma * np.sqrt((1 - alpha) / k)
        grad = 0
        # for i in range(pop_size):
        for i, (inputs, targets) in enumerate(self.loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            if alpha > 0.5:
                # noise = a * np.random.randn(1, len(x))
                noise = a * torch.rand(1, len(x))
            else:
                # noise = a * np.random.randn(1, len(x)) + c * np.random.randn(1, k) @ U.T
                noise = a * torch.rand(1, len(x)) + c * torch.rand(1, k) @ U.T
            noise = noise.reshape(x.shape).cuda()

            # The loss_fn returns the output with the shape equals to the batch size, yet the noise has the shape of 1 * len(x)
            # torch.Size([256]) torch.Size([256]) f(x+noise) f(x-noise) torch.Size([256])
            # torch.Size([558400]) noise.shape

            grad += noise * (loss_fn(x + noise, self.model, self.criterion, inputs, targets) - \
                             loss_fn(x - noise, self.model, self.criterion, inputs, targets))
        return grad / (2 * pop_size * sigma ** 2)

    def ges(self, x_init, loss_fn, lr=0.2, sigma=0.01, k=1, pop_size=1, max_samples=int(1e5)):
        x = deepcopy(x_init)
        total_sample, current_iter = 0, 0
        U, surg_grads = None, []
        xs, ys, ts, errors = [], [], [], []
        while total_sample < max_samples:
            time_st = time.time()
            if current_iter < k:
                g_hat = self.ges_compute_grads(x, loss_fn, U, k, pop_size=pop_size, sigma=sigma, alpha=1)
                surg_grads.append(g_hat)
            else:
                U, _ = np.linalg.qr(np.array(surg_grads).T)
                g_hat = self.ges_compute_grads(x, loss_fn, U, k, pop_size=pop_size, sigma=sigma, alpha=0.5)
                surg_grads.pop(0)
                surg_grads.append(g_hat)
                # sg = loss_fn.compute_gradient(x, bias_coef=1., noise_coef=1.5)[0]
                # surg_grads.append(sg)
            errors.append(np.dot(2*x, g_hat)/(np.linalg.norm(2*x) * np.linalg.norm(g_hat)))
            x -= lr * g_hat
            with torch.no_grad():
                vector_to_parameters(x, self.model.parameters())

            xs.append(2*pop_size)
            ys.append(loss_fn(x))
            ts.append(time.time() - time_st)
            total_sample += pop_size*2
            current_iter += 1
        print('guided es use time :%.2f sec' % np.sum(ts))
        return xs, ys, ts, errors