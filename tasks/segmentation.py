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
from sklearn.metrics import jaccard_score,cohen_kappa_score
import wandb


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
    def __init__(self, device):
        super(CELoss, self).__init__()
        self.criterion = CrossEntropyLoss(reduce=False)
        self.device = device

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
        return loss.mean(axis=1).to(self.device)


class Segmentation_Measure:
    def __init__(self):
        pass
    
    def pixel_accuracy(self, pred, ground_truth):
        with torch.no_grad():
            pred = torch.argmax(pred, dim=1) #dim =1 => over rows
            correct = torch.eq(pred, ground_truth).int()  # the o/p here is 0 , 1
            accuracy = float(correct.sum()) / float(correct.numel())
        return accuracy

    def mIoU(self, pred, ground_truth):
        with torch.no_grad():
            # normalize the output becuse (we have identity activation functions) linear (same output)
            pred         = torch.argmax(pred, dim=1)
            pred         = pred.cpu().contiguous().view(-1).numpy() # make it 1D
            ground_truth = ground_truth.cpu().contiguous().view(-1).numpy() # make it 1D
            MIoU=jaccard_score(ground_truth, pred,average='macro') #'none' per class
        return MIoU

    def kapp_score_Value(self, pred, ground_truth):
        with torch.no_grad():
            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().contiguous().view(-1).numpy() # make it 1D
            ground_truth = ground_truth.cpu().contiguous().view(-1).numpy() # make it 1D
            kapp_score = cohen_kappa_score(ground_truth, ground_truth)

        return kapp_score

class Segmentation(BaseTask):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.load_data()
        self.build_model(self.args)
        self.measures = Segmentation_Measure()
        
        # Log in to your W&B account
        wandb.login()
        wandb.init(
        # Set the project where this run will be logged
        project="Black-Box Defense Segmentation", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{args.wandb_name}",
        # Track hyperparameters and run metadata
        config=args)
        wandb.watch(self.encoder)
        wandb.watch(self.decoder)

        
    def load_data(self):
        print("Load dataset for segmentation task")
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), \
                                        transforms.RandomVerticalFlip(p=0.5), \
                                        transforms.ToTensor()])
        
        test_transform = transforms.Compose([transforms.ToTensor()])

        temp = Cityscapes(self.args.dataset_path, split='train', batch_size=self.args.batch, transform=train_transform)
        self.train_loader = temp.build_data()

        temp = Cityscapes(self.args.dataset_path, split='train', batch_size=self.args.batch, transform=test_transform)
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
                # assert checkpoint['arch'] == args.encoder_arch
                self.encoder = get_architecture(checkpoint['arch'], args.dataset)
                self.encoder.load_state_dict(checkpoint['state_dict'])
            else:
                self.encoder = get_architecture(args.encoder_arch, args.dataset)

            if args.pretrained_decoder:
                checkpoint = torch.load(args.pretrained_decoder)
                # assert checkpoint['arch'] == args.decoder_arch
                self.decoder = get_architecture(checkpoint['arch'], args.dataset)
                self.decoder.load_state_dict(checkpoint['state_dict'])
            else:
                self.decoder = get_architecture(args.decoder_arch, args.dataset)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = get_segmentation_model(device)
        checkpoint = torch.load(args.classifier)
        self.model = get_architecture('unet', args.dataset)
        self.model.load_state_dict(checkpoint)
        self.model.cuda().eval()
        requires_grad_(self.model, False)
    

    def train(self):
        starting_epoch = 0
        logfilename = os.path.join(self.args.outdir, 'log.txt')
        init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttestAcc")
        if self.args.model_type == 'AE_DS':
            if self.args.train_method =='part':
                self.optimizer = build_opt(self.args.optimizer, nn.ModuleList([self.denoiser]))
            elif self.args.train_method =='whole':
                self.optimizer = build_opt(self.args.optimizer, nn.ModuleList([self.encoder, self.denoiser]))
            elif self.args.train_method =='whole_plus':
                self.optimizer = build_opt(self.args.optimizer, nn.ModuleList([self.encoder, self.decoder, self.denoiser]))
            # elif self.args.train_method =='mid':
            #     self.optimizer = build_opt(self.args.optimizer, nn.ModuleList([self.encoder, self.decoder]))
        elif self.args.model_type == 'DS':
            self.optimizer = build_opt(self.args.optimizer, self.denoiser)

        # self.criterion = CrossEntropyLoss(reduction='mean')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = CELoss(device)
        scheduler = StepLR(self.optimizer, step_size=self.args.lr_step_size, gamma=self.args.gamma)
        for epoch in range(starting_epoch, self.args.epochs):
            before = time.time()
            
            if self.args.model_type == 'AE_DS':
                train_loss = self.train_denoiser_with_ae(epoch)
            elif self.args.model_type == 'DS':
                train_loss = self.train_denoiser(epoch)
            _, train_acc = self.eval(self.train_loader)
            test_loss, test_mAcc, test_mIOU = self.eval(self.test_loader)
            
            after = time.time()

            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before, self.args.lr, train_loss, test_loss, train_acc, test_mAcc, test_mIOU))
            
            wandb.log({"train_loss": train_loss, "test_loss": test_loss, "train_acc": train_acc, "test_mAcc": test_mAcc, "test_mIOU": test_mIOU})
            # wandb.log({"loss": loss, "mAcc": mAcc, "mIOU": mIOU})

            scheduler.step(epoch)
            self.args.lr = scheduler.get_lr()[0]
        wandb.finish()

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
        
        # if self.args.train_method == 'mid':
        #     self.encoder.train()
        #     self.decoder.train()

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
            recon = self.encoder(recon, self.decoder)

            if self.args.optimization_method == 'FO':
                recon = self.decoder(recon)
                recon = self.model(recon)
                # loss = self.criterion(recon, targets)
                loss = CrossEntropyLoss()(recon, targets)

                # record loss
                losses.update(loss.item(), inputs.size(0))

            elif self.args.optimization_method == 'ZO':
                recon.requires_grad_(True)
                recon.retain_grad()
                if 'RGE' in self.args.zo_method or 'CGE' in self.args.zo_method:
                    loss = self.es_adapter.run(inputs, recon, targets)
                else:
                    loss = self.es_adapter.run(inputs, targets)

                losses.update(loss.item(), inputs.size(0))
            

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

                if 'RGE' in self.args.zo_method or 'CGE' in self.args.zo_method:
                    loss = self.es_adapter.run(inputs, recon, targets)
                else:
                    loss = self.es_adapter.run(inputs, targets)

                losses.update(loss.item(), inputs.size(0))

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
        mAcc = AverageMeter()
        mIOU = AverageMeter()
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
                # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                acc = self.measures.pixel_accuracy(outputs, targets)
                miou = self.measures.mIoU(outputs, targets)
                
                losses.update(loss_mean.item(), inputs.size(0))
                mAcc.update(acc, inputs.size(0))
                mIOU.update(miou, inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    log = 'Test: [{0}/{1}]\t'' \
                    ''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'' \
                    ''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'' \
                    ''Loss {loss.val:.4f} ({loss.avg:.4f})\t'' \
                    ''mACC {mAcc.val:.3f} ({mAcc.avg:.3f})\t'' \
                    ''mIOU {mIOU.val:.3f} ({mIOU.avg:.3f})\n'.format(
                        i, len(loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, mAcc=mAcc, mIOU=mIOU)

                    print(log)

            return (losses.avg, mAcc.avg, mIOU.avg)

    def test(self):
        print("Test model for segmentation task.")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mAcc = AverageMeter()
        mIOU = AverageMeter()
        end = time.time()

        # switch to eval mode
        self.denoiser.eval()
        if self.args.model_type == 'AE_DS':
            self.encoder.eval()
            self.decoder.eval()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
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
                # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                acc = self.measures.pixel_accuracy(outputs, targets)
                miou = self.measures.mIoU(outputs, targets)
                
                losses.update(loss_mean.item(), inputs.size(0))
                mAcc.update(acc, inputs.size(0))
                mIOU.update(miou, inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    log = 'Test: [{0}/{1}]\t'' \
                    ''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'' \
                    ''Data {data_time.val:.3f} ({data_time.avg:.3f})\t'' \
                    ''Loss {loss.val:.4f} ({loss.avg:.4f})\t'' \
                    ''mACC {mAcc.val:.3f} ({mAcc.avg:.3f})\t'' \
                    ''mIOU {mIOU.val:.3f} ({mIOU.avg:.3f})\n'.format(
                        i, len(self.test_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, mAcc=mAcc)