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
        self.loss = nn.CrossEntropyLoss(size_average=None, reduce=False, reduction='none')
    
    def forward(self, inputs, targets):
        if len(targets.size()) == 1: # batch_size
            targets = targets.unsqueeze(-1)
        elif len(targets.size()) == 2: # batch_size, cls
            targets = targets.argmax(1).unsqueeze(-1).long()
        inputs = inputs.unsqueeze(-1).float()
        return self.loss(inputs, targets)
        
        # acc = []
        # for input, terget in zip(inputs, targets):
        #     acc1 = accuracy(inputs, targets, topk=(1,))
        #     acc.append(torch.tensor(acc1))
        # acc = torch.stack(acc).cuda()
        # # print('===', acc.shape)
        # return acc

class CS_Loss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CosineSimilarity(dim=1, eps=1e-08)
    
    def forward(self, inputs, targets):
        # if len(targets.size()) == 1: # batch_size
        #     targets = targets.unsqueeze(-1)
        # elif len(targets.size()) == 2: # batch_size, cls
        #     targets = targets.argmax(1).unsqueeze(-1).long()
        # inputs = inputs.unsqueeze(-1)
        
        # batch_size, cls
        return self.loss(inputs, targets).mean()

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


class Cls_Loss(nn.Module):
    def __init__(self, lambda_cs=0.5, lambda_mmd=0.5):
        self.ce_loss = CELoss()
        self.cs_loss = CS_Loss()
        self.mmd_loss = MMD_loss()
        self.lambda_cs = lambda_cs
        self.lambda_mmd = lambda_mmd
    
    def forward(self, source, ori_source, target):
        loss = self.ce_loss(source, target) + self.lambda_cs * self.cs_loss(source, ori_source) + \
                self.lambda_mmd * self.mmd_loss(source, ori_source)
        return loss

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
            if self.args.train_method =='part':
                self.optimizer = build_opt(self.args.optimizer, nn.ModuleList([self.denoiser]))
            if self.args.train_method =='whole':
                self.optimizer = build_opt(self.args.optimizer, nn.ModuleList([self.encoder, self.denoiser]))
            if self.args.train_method =='whole_plus':
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
                if self.args.zo_method == 'CGE':
                    print("Warning: Training CGE and use model type DS")
                train_loss = self.train_denoiser(epoch)
            _, train_acc = self.eval(self.train_loader)
            test_loss, test_acc = self.eval(self.test_loader)
            
            after = time.time()

            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before, self.args.lr, train_loss, test_loss, train_acc, test_acc))
            
            scheduler.step(epoch)
            self.args.lr = scheduler.get_lr()[0]

            self.save(epoch, test_acc)

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
            
            def set_ori(self, ori_inputs):
                self.ori_recon = self.classifier(ori_inputs)
            
            def __call__(self, inputs_q, targets):
                recon_q = self.classifier(inputs_q)
                loss = self.criterion(recon_q, self.ori_recon, targets)
                return loss

        if 'RGE' in self.args.zo_method or 'CGE' in self.args.zo_method:
            self.es_adapter = Adapter_RGE_CGE(zo_method=self.args.zo_method, q=self.args.q, criterion=self.criterion, model=self.model, losses=losses, decoder=self.decoder)
        else:
            model = nn.Sequential(self.decoder, self.model)
            self.loss_fn = loss_fn(self.criterion, model)
            self.es_adapter = Adapter(zo_method=self.args.zo_method, subspace=self.args.q, criterion=self.loss_fn, losses=losses, model=self.model)
        
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

            if not 'RGE' in self.args.zo_method and not 'CGE' in self.args.zo_method:
                self.loss_fn.set_ori(inputs)

            if self.args.optimization_method == 'FO':
                recon = self.decoder(recon)
                recon = self.model(recon)
                loss = self.criterion(recon, targets)
                # record loss
                losses.update(loss.item(), inputs.size(0))

            elif self.args.optimization_method == 'ZO':
                recon.requires_grad_(True)
                recon.retain_grad()
                if 'RGE' in self.args.zo_method or 'CGE' in self.args.zo_method:
                    loss = self.es_adapter.run(inputs, recon, targets)
                else:
                    loss = self.es_adapter.run(inputs, recon, targets)
            
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

        class loss_fn:
            def __init__(self, criterion, classifier):
                self.classifier = classifier
                self.criterion = criterion
            
            # def set_target(self, targets):
            #     self.targets = targets
            
            def __call__(self, inputs_q, targets):
                inputs_q_pre = self.classifier(inputs_q)
                loss = self.criterion(inputs_q_pre, targets)
                return loss

        if 'RGE' in self.args.zo_method or 'CGE' in self.args.zo_method:
            self.es_adapter = Adapter_RGE_CGE(zo_method=self.args.zo_method, q=self.args.q, criterion=self.criterion, model=self.model, losses=losses)
        else:
            # self.es_adapter = Adapter(self.args.zo_method, self.args.q, loss_fn(self.criterion, self.model), self.model)
            self.es_adapter = Adapter(zo_method=self.args.zo_method, subspace=self.args.q, criterion=loss_fn(self.criterion, self.model), losses=losses, model=self.model)
        
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

                if 'RGE' in self.args.zo_method or 'CGE' in self.args.zo_method:
                    loss = self.es_adapter.run(inputs, recon, targets)
                else:
                    loss = self.es_adapter.run(inputs, recon, targets)
                    # prev_loss = nn.MSELoss(size_average=None, reduce=None, reduction='none')(inputs, recon)
                    # loss = prev_loss.view(self.args.batch, -1) @ loss.unsqueeze(-1)
                    # loss = torch.sum(loss) / len(loss)

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