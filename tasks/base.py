import torch
import os
from torch.utils.data import DataLoader


class BaseTask:
    def __init__(self) -> None:
        self.encoder = None
        self.decoder = None
        self.denoiser = None
        self.clf = None
        self.criterion = None
        self.optimizer = None
        self.best_acc = 0

    def load_data(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
    
    def train_with_ae(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
    
    def train_ae(loader: DataLoader, encoder: torch.nn.Module, decoder: torch.nn.Module, denoiser: torch.nn.Module, criterion,
          optimizer: Optimizer, epoch: int, noise_sd: float,
          classifier: torch.nn.Module = None):
        """
        Function for training denoiser for one epoch
            :param loader:DataLoader: training dataloader
            :param denoiser:torch.nn.Module: the denoiser being trained
            :param criterion: loss function
            :param optimizer:Optimizer: optimizer used during trainined
            :param epoch:int: the current epoch (for logging)
            :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
            :param classifier:torch.nn.Module=None: a ``freezed'' classifier attached to the denoiser
                                                    (required classifciation/stability objectives), None for denoising objective
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        # switch to train mode
        denoiser.train()

        if args.train_method == 'part':
            encoder.eval()
            decoder.eval()
        if args.train_method == 'whole':
            encoder.train()
            decoder.eval()
        if args.train_method == 'whole_plus':
            encoder.train()
            decoder.train()

        if classifier:
            classifier.eval()

        class loss_fn:
            def __init__(self, criterion, classifier, decoder):
                self.classifier = classifier
                self.decoder = decoder
                self.criterion = criterion
            
            def set_target(self, targets):
                self.targets = targets
            
            def __call__(self, inputs_q):
                inputs_q_pre = self.classifier(self.decoder(inputs_q))
                loss_tmp_plus = self.criterion(inputs_q_pre, self.targets)
                return loss_tmp_plus

        if args.zo_method =='GES':
            med = GES(args.q, 0.01, 0.005)
            loss_fn = loss_fn(criterion, classifier, decoder)
        elif args.zo_method =='SGES':
            med = SGES(args.q, 0.01, 0.005, True)
            loss_fn = loss_fn(criterion, classifier, decoder)
        elif args.zo_method =='CMA_ES':
            med = CMA_ES(args.q, 0.01, 0.005)
            loss_fn = loss_fn(criterion, classifier, decoder)
            
        from tqdm import tqdm
        for i, (inputs, targets) in tqdm(enumerate(loader)):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()
            if args.ground_truth == 'original_output':
                with torch.no_grad():
                    targets = classifier(inputs)
                    targets = targets.argmax(1).detach().clone()

            # augment inputs with noise
            noise = torch.randn_like(inputs, device='cuda') * noise_sd

            recon = denoiser(inputs + noise)
            recon = encoder(recon)

            if args.optimization_method == 'FO':
                recon = decoder(recon)
                recon = classifier(recon)
                loss = criterion(recon, targets)

                # record loss
                losses.update(loss.item(), inputs.size(0))

            elif args.optimization_method == 'ZO':
                recon.requires_grad_(True)
                recon.retain_grad()

                # Obtain the Shape of Inputs (Batch_size x Channel x H x W)
                batch_size = recon.size()[0]
                channel = recon.size()[1]
                h = recon.size()[2]
                w = recon.size()[3]
                d = channel * h * w

                if args.zo_method =='RGE':
                    with torch.no_grad():
                        m, sigma = 0, 100  # mean and standard deviation
                        mu = torch.tensor(args.mu).cuda()
                        q = torch.tensor(args.q).cuda()

                        # Forward Inference (Original)
                        original_pre = classifier(inputs).argmax(1).detach().clone()

                        recon_pre = classifier(decoder(recon))
                        loss_0 = criterion(recon_pre, original_pre)

                        # record original loss
                        loss_0_mean = loss_0.mean()
                        losses.update(loss_0_mean.item(), inputs.size(0))

                        recon_flat_no_grad = torch.flatten(recon, start_dim=1).cuda()
                        grad_est = torch.zeros(batch_size, d).cuda()

                        # ZO Gradient Estimation
                        for k in range(args.q):
                            # Obtain a random direction vector
                            u = torch.normal(m, sigma, size=(batch_size, d))
                            u_norm = torch.norm(u, p=2, dim=1).reshape(batch_size, 1).expand(batch_size, d)    # dim -- careful
                            u = torch.div(u, u_norm).cuda()       # (batch_size, d)

                            # Forward Inference (reconstructed image + random direction vector)
                            recon_q = recon_flat_no_grad + mu * u
                            recon_q = recon_q.view(batch_size, channel, h, w)
                            recon_q_pre = classifier(decoder(recon_q))

                            # Loss Calculation and Gradient Estimation
                            loss_tmp = criterion(recon_q_pre, original_pre)
                            loss_diff = torch.tensor(loss_tmp - loss_0)
                            grad_est = grad_est + (d / q) * u * loss_diff.reshape(batch_size, 1).expand_as(u) / mu

                    recon_flat = torch.flatten(recon, start_dim=1).cuda()
                    grad_est_no_grad = grad_est.detach()

                    # reconstructed image * gradient estimation   <--   g(x) * a
                    loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()

                elif args.zo_method =='CGE':
                    with torch.no_grad():
                        mu = torch.tensor(args.mu).cuda()
                        q = torch.tensor(args.q).cuda()

                        # Forward Inference (Original)
                        original_pre = classifier(inputs).argmax(1).detach().clone()

                        recon_pre = classifier(decoder(recon))     
                        loss_0 = criterion(recon_pre, original_pre)

                        # record original loss
                        loss_0_mean = loss_0.mean()
                        losses.update(loss_0_mean.item(), inputs.size(0))

                        recon_flat_no_grad = torch.flatten(recon, start_dim=1).cuda()
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
                            recon_q_pre_plus = classifier(decoder(recon_q_plus))
                            recon_q_pre_minus = classifier(decoder(recon_q_minus))

                            # Loss Calculation and Gradient Estimation
                            loss_tmp_plus = criterion(recon_q_pre_plus, original_pre)
                            loss_tmp_minus = criterion(recon_q_pre_minus, original_pre)

                            loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
                            grad_est = grad_est + u * loss_diff.reshape(batch_size, 1).expand_as(u) / (2 * mu)

                    recon_flat = torch.flatten(recon, start_dim=1).cuda()
                    grad_est_no_grad = grad_est.detach()

                    # reconstructed image * gradient estimation   <--   g(x) * a
                    loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()

                elif args.zo_method =='CGE_sim':
                    # Generate Coordinate-wise Query Matrix
                    u_flat = torch.zeros(1, args.q, d).cuda()
                    for k in range(d):
                        u_flat[:, k, k] = 1
                    u_flat = u_flat.repeat(1, batch_size, 1).view(batch_size * args.q, d)
                    u = u_flat.view(-1, channel, h, w)

                    with torch.no_grad():
                        mu = torch.tensor(args.mu).cuda()

                        recon_pre = classifier(decoder(recon))  # (batch_size, 10)

                        loss_0 = criterion(recon_pre, targets)  # (batch_size )
                        loss_0_mean = loss_0.mean()
                        losses.update(loss_0_mean.item(), inputs.size(0))

                        # Repeat q times
                        targets = targets.view(batch_size, 1).repeat(1, args.q).view(batch_size * args.q)  # (batch_size * q, )

                        recon_q = recon.repeat((1, args.q, 1, 1)).view(-1, channel, h, w)
                        recon_q_plus = recon_q + mu * u
                        recon_q_minus = recon_q - mu * u

                        # Black-Box Query
                        recon_q_pre_plus = classifier(decoder(recon_q_plus))
                        recon_q_pre_minus = classifier(decoder(recon_q_minus))
                        loss_tmp_plus = criterion(recon_q_pre_plus, targets)
                        loss_tmp_minus = criterion(recon_q_pre_minus, targets)

                        loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
                        grad_est = u_flat * loss_diff.reshape(batch_size * args.q, 1).expand_as(u_flat) / (2 * mu)
                        grad_est = grad_est.view(batch_size, args.q, d).mean(1, keepdim=True).view(batch_size,d)

                    recon_flat = torch.flatten(recon, start_dim=1).cuda()
                    grad_est_no_grad = grad_est.detach()

                    # reconstructed image * gradient estimation   <--   g(x) * a
                    loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()  # l_mean

                # elif args.zo_method =='GES':
                else:
                    recon_pre = classifier(decoder(recon))  # (batch_size, 10)
                    loss_0 = criterion(recon_pre, targets)  # (batch_size )
                    loss_0_mean = loss_0.mean()
                    losses.update(loss_0_mean.item(), inputs.size(0))
                    if recon.shape[0] != args.batch:
                        continue
                    targets_ = targets.view(batch_size, 1).repeat(1, args.q).view(batch_size * args.q)
                    loss_fn.set_target(targets_)
                    grad_est_no_grad, recon_flat = med.run(recon, loss_fn)

                    # reconstructed image * gradient estimation   <--   g(x) * a
                    loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()  # l_mean

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

        return losses.avg

    def save(self, args, test_acc):
        # -----------------  Save the latest model  -------------------
        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': self.denoiser.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(args.outdir, 'denoiser.pth.tar'))

        if args.model_type == 'AE_DS':
            torch.save({
                'epoch': epoch + 1,
                'arch': args.encoder_arch,
                'state_dict': self.encoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, os.path.join(args.outdir, 'encoder.pth.tar'))

            torch.save({
                'epoch': epoch + 1,
                'arch': args.decoder_arch,
                'state_dict': self.decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, os.path.join(args.outdir, 'decoder.pth.tar'))

        # ----------------- Save the best model according to acc -----------------
        if test_acc > self.best_acc:
            self.best_acc = test_acc
        else:
            return

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': self.denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'best_denoiser.pth.tar'))

        if args.model_type == 'AE_DS':
            torch.save({
                'epoch': epoch + 1,
                'arch': args.encoder_arch,
                'state_dict': self.encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'best_encoder.pth.tar'))

            torch.save({
                'epoch': epoch + 1,
                'arch': args.decoder_arch,
                'state_dict': self.decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'best_decoder.pth.tar'))