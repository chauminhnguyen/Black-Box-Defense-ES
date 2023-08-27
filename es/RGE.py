import torch
import numpy as np

class RGE:
    def __init__(self, q, sigma, mu):
        '''
        q: number of samples
        sigma: noise scale
        k: subspace dimensions
        '''
        self.q = q
        self.mu = mu
    
    def run(self, ori_inputs, inputs, loss_fn):
        with torch.no_grad():
            m, sigma = 0, 100  # mean and standard deviation
            mu = torch.tensor(self.mu).cuda()
            q = torch.tensor(self.q).cuda()

            # Forward Inference (Original)
            original_pre = classifier(ori_inputs).argmax(1).detach().clone()

            recon_pre = classifier(inputs)
            loss_0 = loss_fn(recon_pre, original_pre)

            recon_flat_no_grad = torch.flatten(recon, start_dim=1).cuda()
            
            # Obtain a random direction vector
            u = torch.normal(m, sigma, size=(q, batch_size, d))
            u_norm = torch.norm(u, p=2, dim=1).reshape(q, batch_size, 1).expand(q, batch_size, d)    # dim -- careful
            u = torch.div(u, u_norm).cuda()       # (batch_size, d)

            # Forward Inference (reconstructed image + random direction vector)
            recon_q = recon_flat_no_grad + mu * u
            recon_q = recon_q.view(q, batch_size, channel, h, w)
            recon_q_pre = classifier(recon_q)

            # Loss Calculation and Gradient Estimation
            loss_tmp = loss_fn(recon_q_pre, original_pre)
            loss_diff = torch.tensor(loss_tmp - loss_0)
            grad_est = (d / q) * u * (loss_diff.reshape(q, batch_size, 1).expand_as(u)).sum(dim=0) / mu

        recon_flat = torch.flatten(inputs, start_dim=1).cuda()
        grad_est_no_grad = grad_est.detach()
        return grad_est_no_grad, recon_flat