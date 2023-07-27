import torch
import numpy as np
import random


class GES:
    def __init__(self, q, sigma, mu):
        '''
        q: number of samples
        sigma: noise scale
        k: subspace dimensions
        '''
        self.U = None
        self.surg_grads = []
        self.q = q
        self.sigma = sigma
        self.mu = mu
        # self.k = k # subspace dimensions
        self.alpha = 1
    
    def run(self, inputs, loss_fn):
        batch_size = inputs.size()[0]
        channel = inputs.size()[1]
        h = inputs.size()[2]
        w = inputs.size()[3]
        d = channel * h * w

        a = self.sigma * np.sqrt(self.alpha / d)
        c = self.sigma * np.sqrt((1 - self.alpha) / batch_size)

        if self.alpha > 0.5:
            u_flat = a * torch.rand(batch_size, self.q, d)
            self.alpha = 0.5
        else:
            u_flat = a * torch.rand(batch_size, self.q, d).cuda() + c * torch.rand(1, batch_size).cuda() @ self.U.T
        
        u_flat = u_flat.view(batch_size * self.q, d).cuda()
        u = u_flat.view(-1, channel, h, w)

        with torch.no_grad():
            mu = torch.tensor(self.mu).cuda()

            # Repeat q times
            input_q = inputs.repeat((1, self.q, 1, 1)).view(-1, channel, h, w) # batch_size * q, channel, h, w
            input_q_plus = input_q + mu * u
            input_q_minus = input_q - mu * u

            loss_tmp_plus = loss_fn(input_q_plus)
            loss_tmp_minus = loss_fn(input_q_minus)

            loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
            
            grad_est = u_flat * loss_diff.reshape(batch_size * self.q, 1).expand_as(u_flat) / (2 * self.q * self.sigma ** 2)
            grad_est = grad_est.view(batch_size, self.q, d).sum(1, keepdim=True).view(batch_size,d)
            self.surg_grads.pop(0) if self.surg_grads else self.surg_grads
            self.surg_grads.append(grad_est)

            surg_grads_tensor = torch.cat(self.surg_grads, dim=0)
            self.U, _ = torch.linalg.qr(surg_grads_tensor.T)

        recon_flat = torch.flatten(inputs, start_dim=1).cuda()
        grad_est_no_grad = grad_est.detach()

        return grad_est_no_grad, recon_flat


class SGES:
    def __init__(self, q, sigma, mu, auto_alpha):
        self.q = q
        self.mu = mu
        self.sigma = sigma
        self.auto_alpha = auto_alpha
        self.grad_loss, self.random_loss = [], []
        self.alpha = 0.5
        self.U = None
        self.surg_grads = []

    def sges_compute_grads(self, inputs, loss_fn, alpha=0.2):
        batch_size = inputs.size()[0]
        channel = inputs.size()[1]
        h = inputs.size()[2]
        w = inputs.size()[3]
        d = channel * h * w
        grad, global_grad, sub_grad = 0, [], []
        grad_loss, random_loss = [], []

        with torch.no_grad():
            mu = torch.tensor(self.mu).cuda()
            if random.random() < alpha:
                u_flat = self.sigma / np.sqrt(d) * torch.rand(batch_size, self.q, d).cuda() + self.sigma / np.sqrt(batch_size) * torch.rand(1, batch_size).cuda() @ self.U.T
                u_flat = u_flat.view(batch_size * self.q, d).cuda()
                u = u_flat.view(-1, channel, h, w)

                # Repeat q times
                input_q = inputs.repeat((1, self.q, 1, 1)).view(-1, channel, h, w) # batch_size * q, channel, h, w
                input_q_plus = input_q + mu * u
                input_q_minus = input_q - mu * u

                loss_tmp_plus = loss_fn(input_q_plus)
                loss_tmp_minus = loss_fn(input_q_minus)
                loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)

                self.grad_loss.append(torch.min(loss_tmp_plus, loss_tmp_minus))
                sub_grad.append(u_flat * loss_diff.reshape(batch_size * self.q, 1).expand_as(u_flat) / self.sigma ** 2 / 2)
            else:
                u_flat = self.sigma / np.sqrt(batch_size) * torch.rand(batch_size, self.q, d)
                u_flat = u_flat.view(batch_size * self.q, d).cuda()
                u = u_flat.view(-1, channel, h, w)

                # Repeat q times
                input_q = inputs.repeat((1, self.q, 1, 1)).view(-1, channel, h, w) # batch_size * q, channel, h, w
                input_q_plus = input_q + mu * u
                input_q_minus = input_q - mu * u

                loss_tmp_plus = loss_fn(input_q_plus)
                loss_tmp_minus = loss_fn(input_q_minus)
                loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
            
                self.random_loss.append(torch.min(loss_tmp_plus, loss_tmp_minus))
                global_grad.append(u_flat * loss_diff.reshape(batch_size * self.q, 1).expand_as(u_flat) / self.sigma ** 2 / 2)
                
            grad_est = u_flat * loss_diff.reshape(batch_size * self.q, 1).expand_as(u_flat) / (2 * self.q * self.sigma ** 2)
            grad_est = grad_est.view(batch_size, self.q, d).sum(1, keepdim=True).view(batch_size,d)
            
            if len(self.grad_loss) > 0:
                grad_loss_tensor = torch.cat(self.grad_loss, dim=0)
                mean_grad_loss = torch.mean(grad_loss_tensor, axis=0)
            else:
                mean_grad_loss = 10000
            
            if len(self.random_loss) > 0:
                random_loss_tensor = torch.cat(self.random_loss, dim=0)
                mean_random_loss = torch.mean(random_loss_tensor, axis=0)
            else:
                mean_random_loss = 10000

        return grad_est, global_grad, sub_grad, mean_grad_loss, mean_random_loss

    def run(self, inputs, loss_fn):
        batch_size = inputs.size()[0]
        channel = inputs.size()[1]
        h = inputs.size()[2]
        w = inputs.size()[3]
        d = channel * h * w
        
        if self.U == None:
            grad_est, global_grad, sub_grad, mean_grad_loss, mean_random_loss = self.sges_compute_grads(inputs, loss_fn, 0.0)
        else:
            grad_est, global_grad, sub_grad, mean_grad_loss, mean_random_loss = self.sges_compute_grads(inputs, loss_fn, self.alpha)
        
        self.surg_grads.pop(0) if self.surg_grads else self.surg_grads
        self.surg_grads.append(grad_est)
        if self.auto_alpha:
            self.alpha = self.alpha * 1.005 if mean_grad_loss < mean_random_loss else self.alpha / 1.005
            self.alpha = 0.7 if self.alpha > 0.7 else self.alpha
            self.alpha = 0.3 if self.alpha < 0.3 else self.alpha
        
        surg_grads_tensor = torch.cat(self.surg_grads, dim=0)
        self.U, _ = torch.linalg.qr(surg_grads_tensor.T)

        recon_flat = torch.flatten(inputs, start_dim=1).cuda()
        grad_est_no_grad = grad_est.detach()

        return grad_est_no_grad, recon_flat