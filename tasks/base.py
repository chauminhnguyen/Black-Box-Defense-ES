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
    
    def eval(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save(self, epoch, test_acc):
        # -----------------  Save the latest model  -------------------
        torch.save({
            'epoch': epoch + 1,
            'arch': self.args.arch,
            'state_dict': self.denoiser.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(self.args.outdir, 'denoiser.pth.tar'))

        if self.args.model_type == 'AE_DS':
            torch.save({
                'epoch': epoch + 1,
                'arch': self.args.encoder_arch,
                'state_dict': self.encoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, os.path.join(self.args.outdir, 'encoder.pth.tar'))

            torch.save({
                'epoch': epoch + 1,
                'arch': self.args.decoder_arch,
                'state_dict': self.decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, os.path.join(self.args.outdir, 'decoder.pth.tar'))

        # ----------------- Save the best model according to acc -----------------
        if test_acc > self.best_acc:
            self.best_acc = test_acc
        else:
            return

        torch.save({
            'epoch': epoch + 1,
            'arch': self.args.arch,
            'state_dict': self.denoiser.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(self.args.outdir, 'best_denoiser.pth.tar'))

        if self.args.model_type == 'AE_DS':
            torch.save({
                'epoch': epoch + 1,
                'arch': self.args.encoder_arch,
                'state_dict': self.encoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, os.path.join(self.args.outdir, 'best_encoder.pth.tar'))

            torch.save({
                'epoch': epoch + 1,
                'arch': self.args.decoder_arch,
                'state_dict': self.decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, os.path.join(self.args.outdir, 'best_decoder.pth.tar'))