from base import BaseTask
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from train_utils import build_opt
import torch
from architectures import DENOISERS_ARCHITECTURES, get_architecture, get_segmentation_model, IMAGENET_CLASSIFIERS, AUTOENCODER_ARCHITECTURES
from torch.nn import CrossEntropyLoss
import time


class Classification(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    def load_data(self, dataset_name, batch_size, num_workers, pin_memory):
        print("Load dataset for classification task")
        train_dataset = get_dataset(dataset_name, 'train')
        test_dataset = get_dataset(dataset_name, 'test')

        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=pin_memory)
        self.test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                                 num_workers=num_workers, pin_memory=pin_memory)

    def build(self, args):
        print("Building model for classification task")
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

        # c) Classifier / Reconstructor
        if args.train_objective == 'segmentation':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            clf = get_segmentation_model(device)
        
        else:
            checkpoint = torch.load(args.classifier)
            clf = get_architecture(checkpoint['arch'], args.dataset)
            clf.load_state_dict(checkpoint['state_dict'])
        

    def train(self, args):
        print("Training model for classification task")
        optimizer = build_opt(optimizer_method)
        criterion = CrossEntropyLoss(size_average=None, reduce=False, reduction='none').cuda()
        # --------------------- Start Training -------------------------------
        best_acc = 0
        for epoch in range(starting_epoch, args.epochs):
            before = time.time()

            if args.model_type == 'AE_DS':
                train_loss = train_ae(train_loader, encoder, decoder, denoiser, criterion, optimizer, epoch,
                                    args.noise_sd,
                                    clf)
                _, train_acc = test_with_classifier_ae(train_loader, encoder, decoder, denoiser, criterion,
                                                    args.noise_sd,
                                                    args.print_freq, clf)
                test_loss, test_acc = test_with_classifier_ae(test_loader, encoder, decoder, denoiser, criterion,
                                                            args.noise_sd,
                                                            args.print_freq, clf)
            elif args.model_type == 'DS':
                train_loss = train(train_loader, denoiser, criterion, optimizer, epoch, args.noise_sd,
                                clf)
                _, train_acc = test_with_classifier(train_loader, denoiser, criterion, args.noise_sd,
                                                    args.print_freq, clf)
                test_loss, test_acc = test_with_classifier(test_loader, denoiser, criterion,
                                                        args.noise_sd,
                                                        args.print_freq, clf)
            after = time.time()

            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, after - before,
                args.lr, train_loss, test_loss, train_acc, test_acc))


    def test(self):
        print("Testing model for classification task")