from .base import BaseTask
import torch
from architectures import get_segmentation_model
from torch.nn import CrossEntropyLoss
import time
from torch.optim.lr_scheduler import StepLR
from .train_utils import AverageMeter, accuracy, init_logfile, log, requires_grad_, build_opt
from es2.Adapter import Adapter, Adapter_RGE_CGE
from tqdm import tqdm
import os
import torch.nn as nn
from torchvision import transforms
from datasets import Cityscapes
import torch.nn.functional as F
from architectures import get_architecture

class Decoder_Segmentation(nn.Module):
    def __init__(self, decoder, model):
        super(Decoder_Segmentation, self).__init__()
        self.decoder = decoder
        self.model = model
    
    def forward(self, x):
        x = self.decoder(x)
        x = self.model(x)
        return x

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.criterion = CrossEntropyLoss(reduce=False)

    def forward(self, inputs, targets):
        '''
        inputs: (n, c, h, w)
        targets: (n, c, h, w)
        '''
        #flatten label and prediction tensors
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets = targets.view(targets.shape[0], targets.shape[1], -1)
        targets_argmax = targets.argmax(axis=1)
        loss = self.criterion(inputs, targets_argmax)
        return loss.mean(axis=1)

class Segmentation(BaseTask):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.load_data()
        self.build_model(self.args)
        
    def load_data(self):
        print("Load dataset for segmentation task")
        dataset_path = "/content/mmsegmentation/data/cityscapes"
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), \
                                        transforms.RandomVerticalFlip(p=0.5), \
                                        transforms.ToTensor()])
        
        test_transform = transforms.Compose([transforms.ToTensor()])

        temp = Cityscapes(dataset_path, split='train', batch_size=self.args.batch, transform=train_transform)
        self.train_loader = temp.build_data()

        temp = Cityscapes(dataset_path, split='train', batch_size=self.args.batch, transform=test_transform)
        self.test_loader = temp.build_data()

    def build_model(self, args):
        print("Building model for segmentation task")
        
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
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_segmentation_model(device)

    def train(self):
        starting_epoch = 0
        logfilename = os.path.join(self.args.outdir, 'log.txt')
        init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttestAcc")
        if self.args.model_type == 'AE_DS':
            self.optimizer = build_opt(self.args.optimizer, nn.ModuleList([self.encoder, self.decoder, self.denoiser]))
        elif self.args.model_type == 'DS':
            self.optimizer = build_opt(self.args.optimizer, self.denoiser)

        # self.criterion = CrossEntropyLoss(reduction='mean')
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

        print("Training model for segmentation task with auto-encoder.")

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

        class loss_fn:
            def __init__(self, criterion, segmentation_model):
                self.segmentation_model = segmentation_model
                self.criterion = criterion
            
            def set_target(self, targets):
                self.targets = targets
            
            def __call__(self, inputs_q):
                inputs_q_pre = self.segmentation_model(inputs_q)
                # inputs_q_pre = F.one_hot(inputs_q_pre, num_classes=35).permute(0,3,1,2).cuda()
                loss_tmp_plus = self.criterion(inputs_q_pre, self.targets)
                return loss_tmp_plus
        
        if 'RGE' in self.args.zo_method or 'CGE' in self.args.zo_method:
            self.es_adapter = Adapter_RGE_CGE(zo_method=self.args.zo_method, q=self.args.q, criterion=self.criterion, model=self.model, losses=losses, decoder=self.decoder)
        else:
            model = Decoder_Segmentation(decoder=self.decoder, model=self.model)
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
        print("Training model for segmentation task.")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        # switch to train mode
        self.denoiser.train()

        class loss_fn:
            def __init__(self, criterion, segmentation_model):
                self.segmentation_model = segmentation_model
                self.criterion = criterion
            
            def set_target(self, targets):
                self.targets = targets
            
            def __call__(self, inputs_q):
                inputs_q_pre = self.segmentation_model(inputs_q)
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
        print("Evaluate model for segmentation task.")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        # switch to eval mode
        
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
        pass