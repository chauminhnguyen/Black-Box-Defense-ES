from typing import Any
from architectures import DENOISERS_ARCHITECTURES, get_architecture, get_segmentation_model, IMAGENET_CLASSIFIERS, AUTOENCODER_ARCHITECTURES
from datasets import get_dataset, DATASETS, Cityscapes
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code, requires_grad_, measurement
import torch.nn.functional as F

import argparse
from datetime import datetime
import os
import time
import torch
import itertools
from robustness import datasets as dataset_r
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
from torchvision.utils import save_image
from recon_attacks import Attacker, recon_PGD_L2
from es import GES, SGES
from torchvision import transforms
from tasks.classification import Classification


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Training Objective
parser.add_argument('--train_objective', default='classification', type=str,
                    help="The whole model is built for classificaiton / reconstruction",
                    choices=['classification', 'reconstruction', 'segmentation'])
parser.add_argument('--ground_truth', default='original_output', type=str,
                    help="The choice of groundtruth",
                    choices=['original_output', 'labels'])

# Dataset
parser.add_argument('--dataset', type=str, choices=DATASETS)
parser.add_argument('--data_min', default=-2.5090184, type=float, help='minimum value of training data')
parser.add_argument('--data_max', default=3.3369503, type=float, help='maximum value of training data')

parser.add_argument('--batch', default=256, type=int, metavar='N', help='batchsize (default: 256)')
parser.add_argument('--measurement', default=576, type=int, metavar='N', help='the size of measurement for image reconstruction')


# Optimization Method
parser.add_argument('--optimization_method', default='FO', type=str,
                    help="FO: First-Order (White-Box), ZO: Zeroth-Order (Black-box)",
                    choices=['FO', 'ZO', 'ES'])
parser.add_argument('--zo_method', default='RGE', type=str,
                    help="Random Gradient Estimation: RGE, Coordinate-Wise Gradient Estimation: CGE",
                    choices=['RGE', 'CGE', 'CGE_sim', 'GES', 'SGES'])
parser.add_argument('--q', default=192, type=int, metavar='N',
                    help='query direction (default: 20)')
parser.add_argument('--mu', default=0.005, type=float, metavar='N',
                    help='Smoothing Parameter')

# Model type
parser.add_argument('--model_type', default='AE_DS', type=str,
                    help="Denoiser + (AutoEncoder) + classifier/reconstructor",
                    choices=['DS', 'AE_DS'])
parser.add_argument('--arch', type=str, choices=DENOISERS_ARCHITECTURES)
parser.add_argument('--encoder_arch', type=str, default='cifar_encoder', choices=AUTOENCODER_ARCHITECTURES)
parser.add_argument('--decoder_arch', type=str, default='cifar_decoder', choices=AUTOENCODER_ARCHITECTURES)
parser.add_argument('--classifier', default='', type=str,
                    help='path to the classifier used with the `classificaiton`'
                         'or `stability` objectives of the denoiser.')
parser.add_argument('--pretrained-denoiser', default='', type=str, help='path to a pretrained denoiser')
parser.add_argument('--pretrained-encoder', default='', type=str, help='path to a pretrained encoder')
parser.add_argument('--pretrained-decoder', default='', type=str, help='path to a pretrained decoder')

# Model to be trained
parser.add_argument('--train_method', default='whole', type=str,
                    help="*part*: only denoiser parameters would be optimized; *whole*: denoiser and encoder parameters would be optimized, *whole_plus*: denoiser and auto-encoder parameters would be optimized",
                    choices=['part', 'whole', 'whole_plus'])

# Training Setting
parser.add_argument('--outdir', type=str, help='folder to save denoiser and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='SGD, Adam', choices=['SGD', 'Adam'])
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=100,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of noise distribution for data augmentation")
parser.add_argument('--visual_freq', default=1, type=int,
                    metavar='N', help='visualization frequency (default: 5)')

# Parameters for adv examples generation
parser.add_argument('--noise_num', default=10, type=int,
                    help='number of noise for smoothing')
parser.add_argument('--num_steps', default=40, type=int,
                    help='Number of steps for attack')
parser.add_argument('--epsilon', default=512, type=float)

args = parser.parse_args()
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

toPilImage = ToPILImage()


def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Copy code to output directory
    copy_code(args.outdir)
    # pin_memory = (args.dataset == "imagenet")

    # # --------------------- Dataset Loading ----------------------
    # if args.dataset == 'cifar10' or args.dataset == 'stl10' or args.dataset == 'mnist':
    #     train_dataset = get_dataset(args.dataset, 'train')
    #     test_dataset = get_dataset(args.dataset, 'test')

    #     train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
    #                               num_workers=args.workers, pin_memory=pin_memory)
    #     test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
    #                              num_workers=args.workers, pin_memory=pin_memory)
    
    # elif args.dataset == 'cityscapes':
    #     # dataset_path = os.path.join(os.getenv('PT_DATA_DIR', 'datasets'), 'cityscapes')
    #     dataset_path = "/content/mmsegmentation/data/cityscapes"
    #     train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), \
    #                                     transforms.RandomVerticalFlip(p=0.5), \
    #                                     transforms.ToTensor()])
        
    #     test_transform = transforms.Compose([transforms.ToTensor()])

    #     temp = Cityscapes(dataset_path, split='train', batch_size=args.batch, transform=train_transform)
    #     train_loader = temp.build_data()

    #     temp = Cityscapes(dataset_path, split='train', batch_size=args.batch, transform=test_transform)
    #     test_loader = temp.build_data()
        
    
    # elif args.dataset == 'restricted_imagenet':
    #     in_path = '/localscratch2/damondemon/datasets/imagenet'
    #     in_info_path = '/localscratch2/damondemon/datasets/imagenet_info'
    #     in_hier = ImageNetHierarchy(in_path, in_info_path)

    #     superclass_wnid = ['n02084071', 'n02120997', 'n01639765', 'n01662784', 'n02401031', 'n02131653', 'n02484322',
    #                        'n01976957', 'n02159955', 'n01482330']

    #     class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)
    #     custom_dataset = dataset_r.CustomImageNet(in_path, class_ranges)
    #     train_loader, test_loader = custom_dataset.make_loaders(workers=4, batch_size=args.batch)

    # --------------------- Model Loading -------------------------
    # a) Denoiser
    if args.pretrained_denoiser:
        checkpoint = torch.load(args.pretrained_denoiser)
        assert checkpoint['arch'] == args.arch
        denoiser = get_architecture(checkpoint['arch'], args.dataset)
        denoiser.load_state_dict(checkpoint['state_dict'])
    else:
        denoiser = get_architecture(args.arch, args.dataset)

    # b) AutoEncoder
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

    # # c) Classifier / Reconstructor
    # if args.train_objective == 'segmentation':
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     clf = get_segmentation_model(device)
        
    # else:
    #     checkpoint = torch.load(args.classifier)
    #     clf = get_architecture(checkpoint['arch'], args.dataset)
    #     clf.load_state_dict(checkpoint['state_dict'])
    #     clf.cuda().eval()
    #     requires_grad_(clf, False)

    # # --------------------- Model to be trained ------------------------
    
    # if args.optimizer == 'Adam':
    #     if args.train_method =='part':
    #         optimizer = Adam(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     if args.train_method =='whole':
    #         optimizer = Adam(itertools.chain(denoiser.parameters(), encoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    #     if args.train_method =='whole_plus':
    #         optimizer = Adam(itertools.chain(denoiser.parameters(), encoder.parameters(), decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    # elif args.optimizer == 'SGD':
    #     if args.train_method =='part':
    #         optimizer = SGD(denoiser.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #     if args.train_method =='whole':
    #         optimizer = SGD(itertools.chain(denoiser.parameters(), encoder.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #     if args.train_method =='whole_plus':
    #         optimizer = SGD(itertools.chain(denoiser.parameters(), encoder.parameters(), decoder.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    # --------------------- Start Training -------------------------------
    if args.train_objective == 'classification':
        task = Classification(args)
        task.train()

    elif args.train_objective == 'reconstruction':
        pass
    elif args.train_objective == 'segmentation':
        pass

    
        # # -----------------  Save the latest model  -------------------
        # torch.save({
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': denoiser.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }, os.path.join(args.outdir, 'denoiser.pth.tar'))

        # if args.model_type == 'AE_DS':
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'arch': args.encoder_arch,
        #         'state_dict': encoder.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }, os.path.join(args.outdir, 'encoder.pth.tar'))

        #     torch.save({
        #         'epoch': epoch + 1,
        #         'arch': args.decoder_arch,
        #         'state_dict': decoder.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }, os.path.join(args.outdir, 'decoder.pth.tar'))


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def frozen_module(module):
    for param in module.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    main()
