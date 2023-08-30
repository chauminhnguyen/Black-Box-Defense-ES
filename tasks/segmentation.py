from base import BaseTask
import torch
from architectures import get_segmentation_model
from torch.nn import CrossEntropyLoss
import time
from train_utils import AverageMeter, accuracy, init_logfile, log, requires_grad_, build_opt
from es2 import Adapter
from tqdm import tqdm
import os
import torch.nn as nn
from torchvision import transforms
from mmseg.datasets import Cityscapes
import torch.nn.functional as F


class Segmentation(BaseTask):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        
    def load_data(self):
        dataset_path = "/content/mmsegmentation/data/cityscapes"
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), \
                                        transforms.RandomVerticalFlip(p=0.5), \
                                        transforms.ToTensor()])
        
        test_transform = transforms.Compose([transforms.ToTensor()])

        temp = Cityscapes(dataset_path, split='train', batch_size=self.args.batch, transform=train_transform)
        self.train_loader = temp.build_data()

        temp = Cityscapes(dataset_path, split='train', batch_size=self.args.batch, transform=test_transform)
        self.test_loader = temp.build_data()

    def build_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clf = get_segmentation_model(device)

    def train(self):
        starting_epoch = 0
        logfilename = os.path.join(self.args.outdir, 'log.txt')
        init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttestAcc")
        self.optimizer = build_opt(self.args.optimizer_method)
        self.criterion = CrossEntropyLoss(size_average=None, reduce=False, reduction='none').cuda()
        for epoch in range(starting_epoch, self.args.epochs):
            before = time.time()
            
            if self.args.model_type == 'AE_DS':
                train_loss = self.train_denoiser_with_ae(epoch)
            elif self.args.model_type == 'DS':
                train_loss = self.train_denoiser(self.args, epoch)
            _, train_acc = self.eval(self.train_loader)
            test_loss, test_acc = self.eval(self.test_loader)
            
            after = time.time()

            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before, self.args.lr, train_loss, test_loss, train_acc, test_acc))

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
                inputs_q_pre = F.one_hot(inputs_q_pre, num_classes=35).permute(0,3,1,2).cuda()
                loss_tmp_plus = self.criterion(inputs_q_pre, self.targets)
                return loss_tmp_plus

        model = nn.Sequential(self.decoder, self.model)
        self.es_adapter = Adapter(self.args.zo_method, self.args.q, loss_fn(self.criterion, self.model), model)
        
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

                loss = self.es_adapter.run(inputs, recon, self.model)

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