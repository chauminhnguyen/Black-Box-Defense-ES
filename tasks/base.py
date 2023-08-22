import torch
import os


class BaseTask:
    def __init__(self) -> None:
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save(self, epoch):
        # -----------------  Save the latest model  -------------------
        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'denoiser.pth.tar'))

        if args.model_type == 'AE_DS':
            torch.save({
                'epoch': epoch + 1,
                'arch': args.encoder_arch,
                'state_dict': encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'encoder.pth.tar'))

            torch.save({
                'epoch': epoch + 1,
                'arch': args.decoder_arch,
                'state_dict': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'decoder.pth.tar'))

        # ----------------- Save the best model according to acc -----------------
        if test_acc > best_acc:
            best_acc = test_acc
        else:
            return

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'best_denoiser.pth.tar'))

        if args.model_type == 'AE_DS':
            torch.save({
                'epoch': epoch + 1,
                'arch': args.encoder_arch,
                'state_dict': encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'best_encoder.pth.tar'))

            torch.save({
                'epoch': epoch + 1,
                'arch': args.decoder_arch,
                'state_dict': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'best_decoder.pth.tar'))