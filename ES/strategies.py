# Source https://github.com/IJCAI2020-SGES/SGES
import numpy as np
from copy import deepcopy
import time
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

class GES:
    def __init__(self, loader, model, criterion):
        self.loader = loader
        self.model = model
        self.criterion = criterion

    def set_weights(self, x):
        with torch.no_grad():
            vector_to_parameters(x, self.model.parameters())
    
    # Guided-ES framework to estimate gradients
    def ges_compute_grads(self, x, loss_fn, U, k=1, pop_size=1, sigma=0.1, alpha=0.5):
        # Globalspace param
        a = sigma * np.sqrt(alpha / x.shape[0])
        # Subspace param
        c = sigma * np.sqrt((1 - alpha) / k)
        grad = 0
        # for i in range(pop_size):
        for i, (inputs, targets) in tqdm(enumerate(self.loader)):
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
            
            antithetic = loss_fn(x + noise, self.criterion, inputs, targets) - \
                         loss_fn(x - noise, self.criterion, inputs, targets)
            antithetic = antithetic.unsqueeze(1).detach()
            noise = noise.unsqueeze(0).detach()
            g = (antithetic * noise).sum(axis=0)
            grad += g
        return grad / (2 * pop_size * sigma ** 2)

    def ges(self, x_init, loss_fn, lr=0.2, sigma=0.01, k=1, pop_size=1, max_samples=int(1e5)):
        x = deepcopy(x_init)
        total_sample, current_iter = 0, 0
        U, surg_grads = None, []
        errors = []
        while total_sample < max_samples:
            if current_iter < k:
                g_hat = self.ges_compute_grads(x, loss_fn, U, k, pop_size=pop_size, sigma=sigma, alpha=1)
                g_hat_cpu = g_hat.cpu()
                surg_grads.append(g_hat_cpu)
            else:
                surg_grads_tensor = torch.cat(surg_grads, dim=0).unsqueeze(0)
                U, _ = torch.linalg.qr(surg_grads_tensor.T)
                g_hat = self.ges_compute_grads(x, loss_fn, U, k, pop_size=pop_size, sigma=sigma, alpha=0.5)
                g_hat_cpu = g_hat.cpu()
                surg_grads.pop(0)
                surg_grads.append(g_hat_cpu)
                # sg = loss_fn.compute_gradient(x, bias_coef=1., noise_coef=1.5)[0]
                # surg_grads.append(sg)
            
            x_cpu = x.cpu()
            print(x_cpu.max(), x_cpu.min(), x_cpu.mean(), x_cpu.std())
            print(g_hat_cpu.max(), g_hat_cpu.min(), g_hat_cpu.mean(), g_hat_cpu.std())
            errors.append(np.dot(2*x_cpu, g_hat_cpu)/(np.linalg.norm(2*x_cpu) * np.linalg.norm(g_hat_cpu)))
            x -= lr * g_hat
            with torch.no_grad():
                vector_to_parameters(x, self.model.parameters())

            total_sample += pop_size*2
            current_iter += 1
        return errors, x