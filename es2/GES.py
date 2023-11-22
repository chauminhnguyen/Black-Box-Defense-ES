import torch
import numpy as np

class GES:
    def __init__(self, sigma, beta, loss_fn):
        '''
        q: number of samples
        sigma: noise scale
        k: subspace dimensions
        '''
        self.U = None
        # self.surg_grads = []
        # self.q = q
        self.sigma = sigma
        self.beta = beta
        # self.k = k # subspace dimensions
        self.alpha = 1
        self.loss_fn = loss_fn
    
    def run(self, ori_inputs, inputs, targets):
        # inputs shape: (batch, channel, h, w)
        batch_size = ori_inputs.size()[0]
        channel = ori_inputs.size()[1]
        h = ori_inputs.size()[2]
        w = ori_inputs.size()[3]

        n = inputs.size()[1] # Subspace
        k = channel * h * w # Param space

        a = self.sigma * np.sqrt(self.alpha / n)
        c = self.sigma * np.sqrt((1 - self.alpha) / k)

        if self.alpha > 0.5:
            # u_flat
            noise = a * torch.rand(batch_size, n, k)
            self.alpha = 0.5
        else:
            noise = a * torch.rand(batch_size, n).cuda() + c * torch.rand(batch_size, k).cuda() @ self.U.T
        
        # noise shape: (batch, k)
        u = noise
        # noise = noise.view(batch_size * self.q, d).cuda()
        # u = noise.view(-1, channel, h, w)

        with torch.no_grad():
            # mu = torch.tensor(self.mu).cuda()

            # Repeat q times
            # input_q = inputs.repeat((1, self.q, 1, 1)).view(-1, channel, h, w) # batch_size * q, channel, h, w
            
            # input shape (batch, k) +/- noise shape (batch, k)
            input_q_plus = inputs + u
            input_q_minus = inputs - u

            loss_tmp_plus = self.loss_fn(input_q_plus, targets)
            loss_tmp_minus = self.loss_fn(input_q_minus, targets)
            # loss_tmp_plus and loss_tmp_minus shape: (batch, 1)

            loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
            # loss_diff shape: (batch, 1)
            
            # grad_est = noise * loss_diff.reshape(batch_size * self.q, 1).expand_as(u_flat) / (2 * self.q * self.sigma ** 2)
            # grad_est = grad_est.view(batch_size, self.q, d).sum(1, keepdim=True).view(batch_size,d)
            
            grad_ests = torch.bmm(noise, loss_diff)
            # grad_ests shape: (batch, k)
            g_hat = self.beta / (2 * self.sigma**2 * batch_size) * torch.sum(grad_ests, dim=0)
            # grad_est shape: (1, k)
            
            # self.surg_grads.pop(0) if self.surg_grads else self.surg_grads
            # self.surg_grads.append(grad_est)

            # surg_grads_tensor = torch.cat(self.surg_grads, dim=0)
            # self.U, _ = torch.linalg.qr(surg_grads_tensor.T)
            self.U, _ = torch.linalg.qr(grad_ests)

        recon_flat = torch.flatten(inputs, start_dim=1).cuda()
        g_hat_no_grad = g_hat.detach()

        return g_hat_no_grad, recon_flat