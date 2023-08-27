from es import *

class Adapter:
    def __init__(self, zo_method, q, loss_fn) -> None:
        self.q = q
        self.zo_method = zo_method
        if zo_method =='GES':
            self.med = GES(q, 0.01, 0.005)
        elif zo_method =='SGES':
            self.med = SGES(q, 0.01, 0.005, True)
        elif zo_method =='RGE':
            self.med = RGE(q, 0.01, 0.005)
        
        self.loss_fn = loss_fn

    def run(self, inputs, targets, model):
        batch_size = inputs.size()[0]
        channel = inputs.size()[1]
        h = inputs.size()[2]
        w = inputs.size()[3]
        d = channel * h * w

        recon_pre = model(inputs)  # (batch_size, 10)
        # loss_0 = criterion(recon_pre, targets)  # (batch_size )
        # loss_0_mean = loss_0.mean()
        # losses.update(loss_0_mean.item(), inputs.size(0))
        if inputs.shape[0] != batch_size:
            return

        targets_ = targets.view(batch_size, 1).repeat(1, self.q).view(batch_size * self.q)
        self.loss_fn.set_target(targets_)
        grad_est_no_grad, recon_flat = self.med.run(recon_pre, self.loss_fn)

        # reconstructed image * gradient estimation   <--   g(x) * a
        loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()  # l_mean
        return loss