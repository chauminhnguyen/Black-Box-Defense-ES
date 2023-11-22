from es2 import *
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Adapter():
    def __init__(self, zo_method, q, loss_fn, model) -> None:
        self.q = q
        self.mu = 0.005
        self.sigma = 0.01
        self.beta = 0.5
        # self.zo_method = zo_method
        if zo_method =='GES':
            self.med = GES(self.sigma, self.beta, loss_fn)
        elif zo_method =='SGES':
            # self.med = SGES(self.q, self.sigma, self.mu, True)
            self.med = SGES(self.sigma, self.beta, loss_fn, True)
        # elif zo_method =='RGE':
        #     self.med = RGE(self.q, self.sigma, self.mu)
        
        # self.loss_fn = loss_fn
        # self.model = model

    def run(self, ori_inputs, inputs, targets):
        # batch_size = inputs.size()[0]
        
        # recon_pre = self.model(inputs)  # (batch_size, 10)
        # loss_0 = criterion(recon_pre, targets)  # (batch_size )
        # loss_0_mean = loss_0.mean()
        # losses.update(loss_0_mean.item(), inputs.size(0))
        
        # if inputs.shape[0] != batch_size:
        #     return

        # targets_ = targets.view(batch_size, 1).repeat(1, self.q).view(batch_size * self.q)
        # self.loss_fn.set_target(targets_)
        grad_est_no_grad, recon_flat = self.med.run(ori_inputs, inputs, targets)

        # reconstructed image * gradient estimation   <--   g(x) * a
        loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()  # l_mean
        return loss
    

class Adapter_RGE_CGE():
    def __init__(self, zo_method, q, criterion, losses, model, decoder=None) -> None:
        self.zo_method = zo_method
        self.q = q
        self.mu = 0.005
        self.sigma = 0.01

        self.model = model
        self.decoder = decoder
        self.criterion = criterion
        self.losses = losses

    def run(self, ori_inputs, inputs, targets):
        batch_size = inputs.size()[0]
        channel = inputs.size()[1]
        h = inputs.size()[2]
        w = inputs.size()[3]
        d = channel * h * w

        if self.zo_method == 'CGE':
            with torch.no_grad():
                mu = torch.tensor(self.mu).to(DEVICE)
                q = torch.tensor(self.q).to(DEVICE)

                # Forward Inference (Original)
                # original_pre = self.model(ori_inputs).argmax(1).detach().clone()
                original_pre = self.model(ori_inputs).detach().clone()

                if self.decoder is None:
                    recon_pre = self.model(inputs)
                else:
                    recon_pre = self.model(self.decoder(inputs))
                
                loss_0 = self.criterion(recon_pre, original_pre)

                # record original loss
                loss_0_mean = loss_0.mean()
                # self.losses.update(loss_0_mean.item(), inputs.size(0))

                recon_flat_no_grad = torch.flatten(inputs, start_dim=1).to(DEVICE)
                grad_est = torch.zeros(batch_size, d).to(DEVICE)

                # ZO Gradient Estimation
                for k in range(d):
                    # Obtain a direction vector (1-0)
                    u = torch.zeros(batch_size, d).to(DEVICE)
                    u[:, k] = 1

                    # Forward Inference (reconstructed image + random direction vector)
                    recon_q_plus = recon_flat_no_grad + mu * u
                    recon_q_minus = recon_flat_no_grad - mu * u

                    recon_q_plus = recon_q_plus.view(batch_size, channel, h, w)
                    recon_q_minus = recon_q_minus.view(batch_size, channel, h, w)
                    if self.decoder is None:
                        recon_q_pre_plus = self.model(recon_q_plus)
                        recon_q_pre_minus = self.model(recon_q_minus)
                    else:
                        recon_q_pre_plus = self.model(self.decoder(recon_q_plus))
                        recon_q_pre_minus = self.model(self.decoder(recon_q_minus))

                    # Loss Calculation and Gradient Estimation
                    loss_tmp_plus = self.criterion(recon_q_pre_plus, original_pre)
                    loss_tmp_minus = self.criterion(recon_q_pre_minus, original_pre)

                    loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
                    grad_est = grad_est + u * loss_diff.reshape(batch_size, 1).expand_as(u) / (2 * mu)

            recon_flat = torch.flatten(inputs, start_dim=1).to(DEVICE)
            grad_est_no_grad = grad_est.detach()

            # reconstructed image * gradient estimation   <--   g(x) * a
            loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()
            
            # self.losses.update(loss, inputs.size(0))
            return loss
        
        elif self.zo_method == 'CGE_sim':
            # Generate Coordinate-wise Query Matrix
            u_flat = torch.zeros(1, self.q, d).to(DEVICE)
            for k in range(d):
                u_flat[:, k, k] = 1
            u_flat = u_flat.repeat(1, batch_size, 1).view(batch_size * self.q, d)
            u = u_flat.view(-1, channel, h, w)

            with torch.no_grad():
                mu = torch.tensor(self.mu).to(DEVICE)

                recon_pre = self.model(self.decoder(inputs))  # (batch_size, 10)

                loss_0 = self.criterion(recon_pre, targets)  # (batch_size )
                loss_0_mean = loss_0.mean()
                self.losses.update(loss_0_mean.item(), inputs.size(0))

                # Repeat q times
                targets = targets.view(batch_size, 1).repeat(1, self.q).view(batch_size * self.q)  # (batch_size * q, )

                recon_q = inputs.repeat((1, self.q, 1, 1)).view(-1, channel, h, w)
                recon_q_plus = recon_q + mu * u
                recon_q_minus = recon_q - mu * u

                # Black-Box Query
                recon_q_pre_plus = self.model(self.decoder(recon_q_plus))
                recon_q_pre_minus = self.model(self.decoder(recon_q_minus))
                loss_tmp_plus = self.criterion(recon_q_pre_plus, targets)
                loss_tmp_minus = self.criterion(recon_q_pre_minus, targets)

                loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
                grad_est = u_flat * loss_diff.reshape(batch_size * self.q, 1).expand_as(u_flat) / (2 * mu)
                grad_est = grad_est.view(batch_size, self.q, d).mean(1, keepdim=True).view(batch_size,d)

            recon_flat = torch.flatten(inputs, start_dim=1).to(DEVICE)
            grad_est_no_grad = grad_est.detach()

            # reconstructed image * gradient estimation   <--   g(x) * a
            loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()  # l_mean
            return loss
        
        elif self.zo_method == 'RGE':
            with torch.no_grad():
                m, sigma = 0, 100  # mean and standard deviation
                mu = torch.tensor(self.mu).to(DEVICE)
                q = torch.tensor(self.q).to(DEVICE)

                # Forward Inference (Original)
                # original_pre = self.model(ori_inputs).argmax(1).detach().clone()
                original_pre = self.model(ori_inputs).detach().clone()

                if self.decoder is None:
                    recon_pre = self.model(inputs)
                else:
                    recon_pre = self.model(self.decoder(inputs))
                loss_0 = self.criterion(recon_pre, original_pre)

                # record original loss
                loss_0_mean = loss_0.mean()
                self.losses.update(loss_0_mean.item(), inputs.size(0))

                recon_flat_no_grad = torch.flatten(inputs, start_dim=1).to(DEVICE)
                grad_est = torch.zeros(batch_size, d).to(DEVICE)

                # ZO Gradient Estimation
                for k in range(self.q):
                    # Obtain a random direction vector
                    u = torch.normal(m, sigma, size=(batch_size, d))
                    u_norm = torch.norm(u, p=2, dim=1).reshape(batch_size, 1).expand(batch_size, d)    # dim -- careful
                    u = torch.div(u, u_norm).to(DEVICE)       # (batch_size, d)

                    # Forward Inference (reconstructed image + random direction vector)
                    recon_q = recon_flat_no_grad + mu * u
                    recon_q = recon_q.view(batch_size, channel, h, w)
                    if self.decoder is None:
                        recon_q_pre = self.model(recon_q)
                    else:
                        recon_q_pre = self.model(self.decoder(recon_q))

                    # Loss Calculation and Gradient Estimation
                    loss_tmp = self.criterion(recon_q_pre, original_pre)
                    loss_diff = torch.tensor(loss_tmp - loss_0)
                    grad_est = grad_est + (d / q) * u * loss_diff.reshape(batch_size, 1).expand_as(u) / mu

            recon_flat = torch.flatten(inputs, start_dim=1).to(DEVICE)
            grad_est_no_grad = grad_est.detach()

            # reconstructed image * gradient estimation   <--   g(x) * a
            loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()
            return loss