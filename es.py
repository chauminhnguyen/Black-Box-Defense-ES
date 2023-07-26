import torch
import numpy as np

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
            
            grad_est = u_flat * loss_diff.reshape(batch_size * self.q, 1).expand_as(u_flat) / (2 * mu)
            grad_est = grad_est.view(batch_size, self.q, d).sum(1, keepdim=True).view(batch_size,d) / (2 * self.q * self.sigma ** 2)
            self.surg_grads.pop(0) if self.surg_grads else self.surg_grads
            self.surg_grads.append(grad_est)

        surg_grads_tensor = torch.cat(self.surg_grads, dim=0)
        self.U, _ = torch.linalg.qr(surg_grads_tensor.T)

        recon_flat = torch.flatten(inputs, start_dim=1).cuda()
        grad_est_no_grad = grad_est.detach()

        return grad_est_no_grad, recon_flat


class SGES:
    def __init__(self) -> None:
        pass

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
            
            grad_est = u_flat * loss_diff.reshape(batch_size * self.q, 1).expand_as(u_flat) / (2 * mu)
            grad_est = grad_est.view(batch_size, self.q, d).sum(1, keepdim=True).view(batch_size,d) / (2 * self.q * self.sigma ** 2)
            self.surg_grads.pop(0) if self.surg_grads else self.surg_grads
            self.surg_grads.append(grad_est)

        surg_grads_tensor = torch.cat(self.surg_grads, dim=0)
        self.U, _ = torch.linalg.qr(surg_grads_tensor.T)

        recon_flat = torch.flatten(inputs, start_dim=1).cuda()
        grad_est_no_grad = grad_est.detach()

        return grad_est_no_grad, recon_flat