from base import BaseTask
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from train_utils import build_opt
import torch
from architectures import DENOISERS_ARCHITECTURES, get_architecture
from torch.nn import MSELoss
import time
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code, requires_grad_, measurement
from es2 import Adapter
from tqdm import tqdm
import os


class Classification(BaseTask):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def load_data(self, dataset_name, batch_size, num_workers, pin_memory):
        print("Load dataset for classification task")
        train_dataset = get_dataset(dataset_name, 'train')
        test_dataset = get_dataset(dataset_name, 'test')

        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=pin_memory)
        self.test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                                 num_workers=num_workers, pin_memory=pin_memory)

    def build_model(self, args):
        print("Building model for classification task")
        
        if args.pretrained_denoiser:
            checkpoint = torch.load(args.pretrained_denoiser)
            assert checkpoint['arch'] == args.arch
            denoiser = get_architecture(checkpoint['arch'], args.dataset)
            denoiser.load_state_dict(checkpoint['state_dict'])
        else:
            denoiser = get_architecture(args.arch, args.dataset)

        if args.model_type == 'AE_DS':
            if args.pretrained_encoder:
                checkpoint = torch.load(args.pretrained_encoder)
                assert checkpoint['arch'] == args.encoder_arch
                encoder = get_architecture(checkpoint['arch'], args.dataset)
                encoder.load_state_dict(checkpoint['state_dict'])
            else:
                encoder = get_architecture(args.encoder_arch, args.dataset)

            if args.pretrained_decoder:
                checkpoint = torch.load(args.pretrained_decoder)
                assert checkpoint['arch'] == args.decoder_arch
                decoder = get_architecture(checkpoint['arch'], args.dataset)
                decoder.load_state_dict(checkpoint['state_dict'])
            else:
                decoder = get_architecture(args.decoder_arch, args.dataset)
        
        checkpoint = torch.load(args.classifier)
        self.model = get_architecture(checkpoint['arch'], args.dataset)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda().eval()
        requires_grad_(self.model, False)
    
    def train(self, args, train_loader):
        starting_epoch = 0
        logfilename = os.path.join(args.outdir, 'log.txt')
        init_logfile(logfilename,
                     "epoch\ttime\tlr\ttrain_stab_loss\tClean_TestLoss_NoDenoiser\tSmoothed_Clean_TestLoss_NoDenoiser\tClean_TestLoss\tSmoothed_Clean_TestLoss\tNoDenoiser_AdvLoss\tSmoothed_NoDenoiser_AdvLoss\tAdv_Loss\tSmoothed_AdvLoss")
        self.optimizer = build_opt(args.optimizer_method)
        self.criterion = MSELoss(size_average=None, reduce=None, reduction='none').cuda()
        for epoch in range(starting_epoch, args.epochs):
            before = time.time()
            
            if self.args.model_type == 'AE_DS':
                train_loss = self.train_denoiser_with_ae(epoch)
            elif self.args.model_type == 'DS':
                train_loss = self.train_denoiser(args, epoch)
            _, train_acc = self.test_with_classifier(self.train_loader)
            test_loss, test_acc = self.test_with_classifier(self.test_loader)
            
            after = time.time()

            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before, args.lr, train_loss, test_loss, train_acc, test_acc))


    def train_denoiser_with_ae(self, epoch):
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
        n_measurement = self.args.measurement
        mm_min = torch.tensor(self.args.data_min, dtype=torch.long).cuda()
        mm_max = torch.tensor(self.args.data_max, dtype=torch.long).cuda()
        mm_dis = mm_max - mm_min

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()

        # switch to train mode
        self.denoiser.train()

        if self.args.train_method == 'part':
            self.encoder.eval()
            self.decoder.eval()
        if self.args.train_method == 'whole':
            self.encoder.train()
            self.decoder.eval()
        if self.args.train_method == 'whole_plus':
            self.encoder.train()
            self.decoder.train()

        self.model.eval()

        for i, (img_original, _) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # Obtain the Shape of Inputs (Batch_size x Channel x H x W)
            batch_size = img_original.size()[0]
            channel = img_original.size()[1]
            h = img_original.size()[2]
            w = img_original.size()[3]
            d = channel * h * w

            img_original = img_original.cuda()           # input x (batch,  channel, h, w)
            img = img_original.view(batch_size, d).cuda()

            if i ==0:
                a = measurement(n_measurement, d)    # Measurement Matrix
                print("-----------------------training--------------------")
                print(a[0, :])

            img = torch.mm(img, a)     # y = A^T x
            img = torch.mm(img, a.t())
            img = img.view(batch_size, channel, h, w)
            img = img.float()
            img = (img - mm_min) / mm_dis

            #augment inputs with noise
            noise = torch.randn_like(img, device='cuda') * self.args.noise_sd
            recon = self.encoder(self.denoiser(img + noise))

            if self.args.optimization_method == 'FO':
                recon = self.model(self.decoder(recon))
                stab_loss = self.criterion(recon, img_original)

                # record original loss
                stab_loss_mean = stab_loss.mean()
                losses.update(stab_loss_mean.item(), img_original.size(0))

                # compute gradient and do step
                self.optimizer.zero_grad()
                stab_loss_mean.backward()
                self.optimizer.step()

            elif self.args.optimization_method == 'ZO':
                recon.requires_grad_(True)
                recon.retain_grad()

                with torch.no_grad():
                    mu = torch.tensor(args.mu).cuda()
                    q = torch.tensor(args.q).cuda()

                    # Forward Inference (Original)
                    original_recon = self.model(img)

                    recon_test = self.model(decoder(recon))
                    loss_0 = criterion(recon_test, original_recon)
                    # record original loss
                    loss_0_mean = loss_0.mean()
                    losses.update(loss_0_mean.item(), img_original.size(0))

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
                        recon_q_pre_plus = recon_net(self.decoder(recon_q_plus))
                        recon_q_pre_minus = recon_net(self.decoder(recon_q_minus))

                        # Loss Calculation and Gradient Estimation
                        loss_tmp_plus = self.criterion(recon_q_pre_plus, original_recon)
                        loss_tmp_minus = self.criterion(recon_q_pre_minus, original_recon)

                        loss_diff = torch.tensor(loss_tmp_plus - loss_tmp_minus)
                        loss_diff = loss_diff.mean(3, keepdim=True).mean(2, keepdim=True)
                        grad_est = grad_est + u * loss_diff.reshape(batch_size, 1).expand_as(u) / (2 * mu)

                recon_flat = torch.flatten(recon, start_dim=1).cuda()
                grad_est_no_grad = grad_est.detach()

                # reconstructed image * gradient estimation   <--   g(x) * a
                loss = torch.sum(recon_flat * grad_est_no_grad, dim=-1).mean()  # l_mean
                # compute gradient and do step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                log = 'Epoch: [{0}][{1}/{2}]\t'' \
                ''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'' \
                ''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'' \
                ''Stab_Loss {stab_loss.val:.4f} ({stab_loss.avg:.4f})''\n'.format(
                    epoch, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, stab_loss = losses)

                print(log)

        return losses.avg
    
    def train_denoiser(self, epoch):
        print("Training model for classification task")
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

        print("Training model for classification task.")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        # switch to train mode
        self.denoiser.train()
        self.classifier.eval()

        class loss_fn:
            def __init__(self, criterion, classifier):
                self.classifier = classifier
                self.criterion = criterion
            
            def set_target(self, targets):
                self.targets = targets
            
            def __call__(self, inputs_q):
                inputs_q_pre = self.classifier(inputs_q)
                loss_tmp_plus = self.criterion(inputs_q_pre, self.targets)
                return loss_tmp_plus

        self.es_adapter = Adapter(self.args.zo_method, self.args.q, loss_fn(self.criterion, self.model))
        
        for i, (inputs, targets) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()
            if self.args.ground_truth == 'original_output':
                with torch.no_grad():
                    targets = self.model(inputs)
                    targets = targets.argmax(1).detach().clone()

            noise = torch.randn_like(inputs, device='cuda') * self.args.noise_sd
            recon = self.denoiser(inputs + noise)

            if self.args.optimization_method == 'FO':
                recon = self.model(recon)
                loss = self.criterion(recon, targets)

                # record loss
                losses.update(loss.item(), inputs.size(0))

            elif self.args.optimization_method == 'ZO':
                recon.requires_grad_(True)
                recon.retain_grad()

                loss = self.es_adapter.run(recon, self.model)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # TODO: create the log file
            if i % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
        return losses.avg
    
    def test_with_classifier(self, loader):
        """
        A function to test the classification performance of a denoiser when attached to a given classifier
            :param loader:DataLoader: test dataloader
            :param denoiser:torch.nn.Module: the denoiser
            :param criterion: the loss function (e.g. CE)
            :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
            :param print_freq:int: the frequency of logging
            :param classifier:torch.nn.Module: the classifier to which the denoiser is attached
        """

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        # switch to eval mode
        self.model.eval()
        if self.denoiser:
            self.denoiser.eval()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                # measure data loading time
                data_time.update(time.time() - end)

                inputs = inputs.cuda()
                targets = targets.cuda()

                # augment inputs with noise
                inputs = inputs + torch.randn_like(inputs, device='cuda') * self.args.noise_sd

                if self.denoiser is not None:
                    inputs = self.denoiser(inputs)
                # compute output
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss_mean = loss.mean()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss_mean.item(), inputs.size(0))
                top1.update(acc1.item(), inputs.size(0))
                top5.update(acc5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    log = 'Test: [{0}/{1}]\t'' \
                    ''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'' \
                    ''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'' \
                    ''Loss {loss.val:.4f} ({loss.avg:.4f})\t'' \
                    ''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'' \
                    ''Acc@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                        i, len(loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5)

                    print(log)
            return (losses.avg, top1.avg)

    def test(self):
        print("Testing model for classification task")