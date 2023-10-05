from .base import BaseTask
from torch.utils.data import DataLoader
from datasets import get_dataset
import torch
from torch.optim.lr_scheduler import StepLR
from architectures import get_architecture
from torch.nn import CrossEntropyLoss
import time
from .train_utils import AverageMeter, accuracy, init_logfile, log, requires_grad_, build_opt
from es2.Adapter import Adapter, Adapter_RGE_CGE
from tqdm import tqdm
import os
import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        inputs = inputs.unsqueeze(-1).float()
        targets = targets.argmax(1).unsqueeze(-1).long()
        return self.loss(inputs, targets)


class Classification(BaseTask):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.load_data(args.dataset, args.batch, args.workers)
        self.build_model(self.args)

    def load_data(self, dataset_name, batch_size, num_workers):
        print("Load dataset for classification task")
        train_dataset = get_dataset(dataset_name, 'train')
        test_dataset = get_dataset(dataset_name, 'test')

        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                                 num_workers=num_workers)

    def build_model(self, args):
        print("Building model for classification task")
        
        if args.pretrained_denoiser:
            checkpoint = torch.load(args.pretrained_denoiser)
            assert checkpoint['arch'] == args.arch
            self.denoiser = get_architecture(checkpoint['arch'], args.dataset)
            self.denoiser.load_state_dict(checkpoint['state_dict'])
        else:
            self.denoiser = get_architecture(args.arch, args.dataset)

        if args.model_type == 'AE_DS':
            if args.pretrained_encoder:
                checkpoint = torch.load(args.pretrained_encoder)
                assert checkpoint['arch'] == args.encoder_arch
                self.encoder = get_architecture(checkpoint['arch'], args.dataset)
                self.encoder.load_state_dict(checkpoint['state_dict'])
            else:
                self.encoder = get_architecture(args.encoder_arch, args.dataset)

            if args.pretrained_decoder:
                checkpoint = torch.load(args.pretrained_decoder)
                assert checkpoint['arch'] == args.decoder_arch
                self.decoder = get_architecture(checkpoint['arch'], args.dataset)
                self.decoder.load_state_dict(checkpoint['state_dict'])
            else:
                self.decoder = get_architecture(args.decoder_arch, args.dataset)
        
        checkpoint = torch.load(args.classifier)
        self.model = get_architecture(checkpoint['arch'], args.dataset)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda().eval()
        requires_grad_(self.model, False)
    
    def train(self):
        starting_epoch = 0
        logfilename = os.path.join(self.args.outdir, 'log.txt')
        init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttestAcc")

        if self.args.model_type == 'AE_DS':
            self.optimizer = build_opt(self.args.optimizer, nn.ModuleList([self.encoder, self.decoder, self.denoiser]))
        elif self.args.model_type == 'DS':
            self.optimizer = build_opt(self.args.optimizer, self.denoiser)

        # self.criterion = CrossEntropyLoss(size_average=None, reduce=False, reduction='none').cuda()
        self.criterion = CELoss()
        scheduler = StepLR(self.optimizer, step_size=self.args.lr_step_size, gamma=self.args.gamma)
        for epoch in range(starting_epoch, self.args.epochs):
            before = time.time()
            
            if self.args.model_type == 'AE_DS':
                train_loss = self.train_denoiser_with_ae(epoch)
            elif self.args.model_type == 'DS':
                train_loss = self.train_denoiser(epoch)
            _, train_acc = self.eval(self.train_loader)
            test_loss, test_acc = self.eval(self.test_loader)
            
            after = time.time()

            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before, self.args.lr, train_loss, test_loss, train_acc, test_acc))
            
            scheduler.step(epoch)
            self.args.lr = scheduler.get_lr()[0]

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

        print("Training model for classification task with auto-encoder.")

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

        if 'RGE' in self.args.zo_method or 'CGE' in self.args.zo_method:
            self.es_adapter = Adapter_RGE_CGE(zo_method=self.args.zo_method, q=self.args.q, criterion=self.criterion, model=self.model, losses=losses, decoder=self.decoder)
        else:
            model = nn.Sequential(self.decoder, self.model)
            self.es_adapter = Adapter(self.args.zo_method, self.args.q, loss_fn(self.criterion, model), self.model)
        
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
            recon = self.encoder(recon)

            if self.args.optimization_method == 'FO':
                recon = self.decoder(recon)
                recon = self.model(recon)
                loss = self.criterion(recon, targets)
                # record loss
                losses.update(loss.item(), inputs.size(0))

            elif self.args.optimization_method == 'ZO':
                recon.requires_grad_(True)
                recon.retain_grad()
                if self.args.zo_method == 'RGE' or self.args.zo_method == 'CGE':
                    loss = self.es_adapter.run(inputs, recon, targets)
                else:
                    loss = self.es_adapter.run(inputs, targets)
            
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
    
    def train_denoiser(self, epoch):
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
        self.model.eval()

        class loss_fn():
            def __init__(self, criterion, classifier):
                self.classifier = classifier
                self.criterion = criterion
            
            def set_target(self, targets):
                self.targets = targets
            
            def __call__(self, inputs_q):
                inputs_q_pre = self.classifier(inputs_q)
                loss_tmp_plus = self.criterion(inputs_q_pre, self.targets)
                return loss_tmp_plus

        if 'RGE' in self.args.zo_method or 'CGE' in self.args.zo_method:
            self.es_adapter = Adapter_RGE_CGE(zo_method=self.args.zo_method, q=self.args.q, criterion=self.criterion, model=self.model, losses=losses)
        else:
            self.es_adapter = Adapter(self.args.zo_method, self.args.q, loss_fn(self.criterion, self.model), self.model)
        
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

                if self.args.zo_method == 'RGE' or self.args.zo_method == 'CGE':
                    loss = self.es_adapter.run(inputs, recon, targets)
                else:
                    loss = self.es_adapter.run(inputs, targets)

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
    
    def eval(self, loader):
        """
        A function to test the classification performance of a denoiser when attached to a given classifier
            :param loader:DataLoader: test dataloader
            :param denoiser:torch.nn.Module: the denoiser
            :param criterion: the loss function (e.g. CE)
            :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
            :param print_freq:int: the frequency of logging
            :param classifier:torch.nn.Module: the classifier to which the denoiser is attached
        """
        print("Evaluate model for classification task.")
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
        if self.args.model_type == 'AE_DS':
            self.encoder.eval()
            self.decoder.eval()

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
                if self.args.model_type == 'AE_DS':
                    inputs = self.encoder(inputs)
                    inputs = self.decoder(inputs)
                    
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
    
    def get_model(self):
        return self.model