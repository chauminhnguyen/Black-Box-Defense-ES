import torch
import numpy as np

class GES:
    def __init__(self, k, sigma, beta, loss_fn, model, decoder=None):
        '''
        q: number of samples
        sigma: noise scale
        k: subspace dimensions
        '''
        self.U = None
        self.surg_grads = []
        self.k = 100
        self.sigma = sigma
        self.beta = beta
        self.alpha = 1
        self.loss_fn = loss_fn
        self.remain = None
        self.cur_iter = 0
        self.P = k
        self.m, self.sigma = 0, 100  # mean and standard deviation
        self.model = model
        self.decoder = decoder
    
    def run(self, inputs, targets):
        # inputs shape: (batch, channel, h, w)
        batch_size = inputs.size()[0]
        channel = inputs.size()[1]
        h = inputs.size()[2]
        w = inputs.size()[3]

        n = channel * h * w # Param space

        a = self.sigma * np.sqrt(self.alpha / self.k)
        c = self.sigma * np.sqrt((1 - self.alpha) / n)

        # if self.alpha > 0.5:
        noise = []
        # for i in self.P:
        if self.cur_iter < self.k:
            # u = torch.normal(self.m, self.sigma, size=(self.P, batch_size, n))
            # u_norm = torch.norm(u, p=2, dim=2).reshape(self.P, batch_size, 1).expand(self.P, batch_size, n)    # dim -- careful
            # u = torch.div(u, u_norm).cuda()       # (batch_size, d)
            u = torch.zeros(self.P, batch_size, n).cuda()
            for i in range(self.P):
                u[i, :, i] = 1

            noise.append(u)
            # self.alpha = 0.5
            self.cur_iter += 1
        else:
            self.alpha = 0.5
            self.U, _ = torch.linalg.qr(torch.stack(self.surg_grads).T)
            
            # u = torch.normal(self.m, self.sigma, size=(self.P, batch_size, n))
            # u_norm = torch.norm(u, p=2, dim=2).reshape(self.P, batch_size, 1).expand(self.P, batch_size, n)    # dim -- careful
            # u_n = torch.div(u, u_norm).cuda()       # (batch_size, d)

            # u = torch.normal(self.m, self.sigma, size=(self.P, batch_size, self.k))
            # u_norm = torch.norm(u, p=2, dim=2).reshape(self.P, batch_size, 1).expand(self.P, batch_size, self.k)    # dim -- careful
            # u_k = torch.div(u, u_norm).cuda()       # (batch_size, d)

            u_n = torch.zeros(self.P, batch_size, n).cuda()
            for i in range(self.P):
                u_n[i, :, i] = 1
            
            u_k = torch.zeros(self.P, batch_size, self.k).cuda()
            for i in range(self.k):
                u_k[i, :, i] = 1

            noise.append(a * u_n + c * u_k @ self.U.T)
            # noise.append(a * torch.rand(self.P, batch_size, n).cuda() + c * torch.rand(self.P, batch_size, self.k).cuda() @ self.U.T)
        
        noise = torch.cat(noise)
        # noise shape: (P, batch, k)

        u = noise.view(self.P, batch_size, channel, h, w)
        # u shape: (P, batch, channel, h, w)


        with torch.no_grad():
            # input shape (P - repeat, batch, channel, h, w) +/- noise shape (P, batch, channel, h, w)
            input_q_plus_arr = inputs.repeat(self.P, 1, 1, 1, 1) + 0.005 * u
            input_q_minus_arr = inputs.repeat(self.P, 1, 1, 1, 1) - 0.005 * u
            
            loss_diff = []
            for input_q_plus, input_q_minus in zip(input_q_plus_arr, input_q_minus_arr):
                if self.decoder is None:
                    recon_q_pre_plus = self.model(input_q_plus)
                    recon_q_pre_minus = self.model(input_q_minus)
                else:
                    recon_q_pre_plus = self.model(self.decoder(input_q_plus))
                    recon_q_pre_minus = self.model(self.decoder(input_q_minus))
                
                loss_tmp_plus = self.loss_fn(recon_q_pre_plus, targets)
                loss_tmp_minus = self.loss_fn(recon_q_pre_minus, targets)
                # loss_tmp_plus and loss_tmp_minus shape: (P, batch, 1)
                loss_diff.extend(loss_tmp_plus - loss_tmp_minus)
                # loss_diff shape: (P, batch, 1)
            loss_diff = torch.stack(loss_diff)
            grad_ests = torch.bmm(noise.view(-1, n).unsqueeze(-1), loss_diff.unsqueeze(-1)).squeeze(-1)
            # grad_ests shape: (P, batch, k)
            g_hat = self.beta / (2 * self.sigma**2 * batch_size) * torch.sum(grad_ests, dim=0)
            # g_hat shape: (batch, k)
            
            if len(self.surg_grads) >= self.k:
                self.surg_grads.pop(0)
            self.surg_grads.append(g_hat)

        return g_hat