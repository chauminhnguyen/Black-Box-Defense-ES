from es2 import *

class Adapter():
    def __init__(self, zo_method, q, loss_fn, model) -> None:
        self.q = q
        self.mu = 0.005
        self.sigma = 0.01
        self.zo_method = zo_method
        if zo_method =='GES':
            self.med = GES(self.q, self.sigma, self.mu)
        elif zo_method =='SGES':
            self.med = SGES(self.q, self.sigma, self.mu, True)
        # elif zo_method =='RGE':
        #     self.med = RGE(self.q, self.sigma, self.mu)
        
        self.loss_fn = loss_fn
        self.model = model

    def run1(self, inputs, ori_inputs):
        batch_size = inputs.size()[0]
        channel = inputs.size()[1]
        h = inputs.size()[2]
        w = inputs.size()[3]
        d = channel * h * w

        if self.zo_method == 'CGE':
            with torch.no_grad():
                mu = torch.tensor(self.mu).cuda()
                q = torch.tensor(self.q).cuda()

                # Forward Inference (Original)
                loss_0 = self.loss_fn(inputs, ori_inputs)

                # # record original loss
                # loss_0_mean = loss_0.mean()
                # losses.update(loss_0_mean.item(), inputs.size(0))

                recon_flat_no_grad = torch.flatten(inputs, start_dim=1).cuda()
                grad_est = torch.zeros(batch_size, d).cuda()

                # ZO Gradient Estimation
                for k in range(d):
                    # Obtain a direction vector (1-0)
                    u = torch.zeros(batch_size, d).cuda()
                    u[:, k] = 1

                    # Forward Inference (reconstructed image + random direction vector)
                    recon_q_plus = recon_flat_no_grad + mu * u
                    recon_q_minus = recon_flat_no_grad - mu * u

                    recon_q_plus = recon_q_plus.view(batch_size, channel, h, w)
                    recon_q_minus = recon_q_minus.view(batch_size, channel, h, w)

                    # Loss Calculation and Gradient Estimation
                    loss_tmp_plus = self.loss_fn(recon_q_plus, ori_inputs)
                    loss_tmp_minus = self.loss_fn(recon_q_minus, ori_inputs)

                    loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
                    grad_est = grad_est + u * loss_diff.reshape(batch_size, 1).expand_as(u) / (2 * mu)

            recon_flat = torch.flatten(inputs, start_dim=1).cuda()
            grad_est_no_grad = grad_est.detach()

            # reconstructed image * gradient estimation   <--   g(x) * a
            loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()
            return loss
        elif self.zo_method == 'RGE':
            with torch.no_grad():
                m, sigma = 0, 100  # mean and standard deviation
                mu = torch.tensor(self.mu).cuda()
                q = torch.tensor(self.q).cuda()

                # Forward Inference (Original)
                loss_0 = self.loss_fn(inputs, ori_inputs)

                recon_flat_no_grad = torch.flatten(inputs, start_dim=1).cuda()
                grad_est = torch.zeros(batch_size, d).cuda()

                # ZO Gradient Estimation
                for k in range(self.q):
                    # Obtain a random direction vector
                    u = torch.normal(m, sigma, size=(batch_size, d))
                    u_norm = torch.norm(u, p=2, dim=1).reshape(batch_size, 1).expand(batch_size, d)    # dim -- careful
                    u = torch.div(u, u_norm).cuda()       # (batch_size, d)

                    # Forward Inference (reconstructed image + random direction vector)
                    recon_q = recon_flat_no_grad + mu * u
                    recon_q = recon_q.view(batch_size, channel, h, w)

                    # Loss Calculation and Gradient Estimation
                    loss_tmp = self.loss_fn(recon_q, ori_inputs)
                    loss_diff = torch.tensor(loss_tmp - loss_0)
                    grad_est = grad_est + (d / q) * u * loss_diff.reshape(batch_size, 1).expand_as(u) / mu

            recon_flat = torch.flatten(inputs, start_dim=1).cuda()
            grad_est_no_grad = grad_est.detach()

            # reconstructed image * gradient estimation   <--   g(x) * a
            loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()
            return loss

    def run2(self, inputs, targets):
        batch_size = inputs.size()[0]
        channel = inputs.size()[1]
        h = inputs.size()[2]
        w = inputs.size()[3]
        d = channel * h * w
        
        # recon_pre = self.model(inputs)  # (batch_size, 10)
        # loss_0 = criterion(recon_pre, targets)  # (batch_size )
        # loss_0_mean = loss_0.mean()
        # losses.update(loss_0_mean.item(), inputs.size(0))
        if inputs.shape[0] != batch_size:
            return

        targets_ = targets.view(batch_size, 1).repeat(1, self.q).view(batch_size * self.q)
        self.loss_fn.set_target(targets_)
        grad_est_no_grad, recon_flat = self.med.run(inputs, self.loss_fn)

        # reconstructed image * gradient estimation   <--   g(x) * a
        loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()  # l_mean
        return loss