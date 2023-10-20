
from torch import nn
# from torchvision.datasets import MNIST
import os
import torch

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5), (0.5))
# ])
#
# dataset = MNIST('./data', transform=img_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class MNIST_CAE(nn.Module):
    def __init__(self):
        super(MNIST_CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # b, 16, 28, 28
            nn.ReLU(True),

            nn.Conv2d(16, 32, 3, stride=3, padding=1), # b, 32, 10, 10
            nn.ReLU(True),

            nn.Conv2d(32, 16, 3, stride=3, padding=1), # b, 16, 4, 4
            nn.ReLU(True),

            nn.Conv2d(16, 8, 3, stride=3, padding=1),  # b, 8, 2, 2
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MNIST_Dim_Encoder(nn.Module):
    def __init__(self):
        super(MNIST_Dim_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, 4, stride=2, padding=1),      #(3, 14, 14)
            nn.ReLU(),
            nn.Conv2d(3, 12, 5, stride=3, padding=0),     #(12, 4, 4)
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class MNIST_Dim_Decoder(nn.Module):
    def __init__(self):
        super(MNIST_Dim_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(12, 3, 5, stride=3, padding=0),   #(3, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 4, stride=2, padding=1),   #(1, 28, 28)
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class CelebA_CAE(nn.Module):
    def __init__(self):
        super(CelebA_CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CIFAR_CAE(nn.Module):
    def __init__(self):
        super(CIFAR_CAE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),            # [batch, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 48, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			#  nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class STL_Encoder(nn.Module):
    def __init__(self):
        super(STL_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 18, 5, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(18, 72, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(72, 144, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(144, 288, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(288, 576, 2, stride=2, padding=0),  # [batch, 576, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class STL_Decoder(nn.Module):
    def __init__(self):
        super(STL_Decoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 288, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(288, 144, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(144, 72, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(72, 18, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(18, 3, 5, stride=3, padding=1),  # [batch, 12, 16, 16]
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class ImageNet_Encoder_15552(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_15552, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 72, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(72, 144, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(144, 432, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(432, 864, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(864, 1728, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_15552(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_15552, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1728, 864, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(864, 432, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(432, 144, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(144, 72, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(72, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded




class ImageNet_Encoder_1152(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_1152, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(96, 192, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(192, 384, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU(),
            nn.Conv2d(384, 1152, 3, stride=3, padding=0),        # [batch, 864, 1, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_1152(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_1152, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1152, 384, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(96,48, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class ImageNet_Encoder_1728(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_1728, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 36, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(36, 72, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(72, 144, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(144, 288, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(288, 576, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU(),
            nn.Conv2d(576, 1728, 3, stride=3, padding=0),        # [batch, 864, 1, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_1728(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_1728, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1728, 576, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(576, 288, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(288, 144, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(144,72, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(72, 36, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(36, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class ImageNet_Encoder_2304(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_2304, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 48, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(96, 192, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(192, 384, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(384, 768, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU(),
            nn.Conv2d(768, 2304, 3, stride=3, padding=0),        # [batch, 864, 1, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_2304(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_2304, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2304, 768, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(768, 384, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(192,96, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(48, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class ImageNet_Encoder_3456(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_3456, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 48, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(96, 288, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(288, 576, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(576, 1152, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU(),
            nn.Conv2d(1152, 3456, 3, stride=3, padding=0),        # [batch, 864, 1, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_3456(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_3456, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3456, 1152, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(1152, 576, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(576, 288, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(288,96, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(48, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class TinyImageNet_Encoder(nn.Module):
    def __init__(self):
        super(TinyImageNet_Encoder, self).__init__()
        # Input size: [batch, 3, 64, 64]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(6, 24, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),

        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class TinyImageNet_Decoder(nn.Module):
    def __init__(self):
        super(TinyImageNet_Decoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 6, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 2, stride=2, padding= 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded

class TinyImageNet_Encoder_768(nn.Module):
    def __init__(self):
        super(TinyImageNet_Encoder_768, self).__init__()
        # Input size: [batch, 3, 64, 64]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0), # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 2, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # nn.Conv2d(24, 48, 4, stride=2, padding=1),
            # nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),  # [batch, 96, 2, 2]
            nn.ReLU(),
            nn.Conv2d(96, 192, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),

        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class TinyImageNet_Decoder_768(nn.Module):
    def __init__(self):
        super(TinyImageNet_Decoder_768, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 2, stride=2, padding=0),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded



class Cifar_Encoder_48(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_48, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(6, 12, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(24, 48, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class Cifar_Decoder_48(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_48, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(12, 6, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded




class Cifar_Encoder_96(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_96, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(48, 96, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class Cifar_Decoder_96(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_96, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded




class Cifar_Encoder_192_24(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_192_24, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(96, 192, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class Cifar_Decoder_192_24(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_192_24, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded




class Cifar_Encoder_192(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_192, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(96, 192, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class Cifar_Decoder_192(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_192, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded



class Cifar_Encoder_384(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_384, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),          # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 96, 2, 2]
        return encoded


class Cifar_Decoder_384(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_384, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Cifar_Encoder_768_32(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_768_32, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			#nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            #nn.ReLU(),
 			#nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             #nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)          # Output Size: [batch, 48, 4, 4]
        return encoded


class Cifar_Decoder_768_32(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_768_32, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             #nn.ReLU(),
			#nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            #nn.ReLU(),
			nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Cifar_Encoder_768_24(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_768_24, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            #nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            #nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
 			#nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             #nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)          # Output Size: [batch, 48, 4, 4]
        return encoded


class Cifar_Decoder_768_24(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_768_24, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             #nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Cifar_Encoder_1536(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_1536, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            #nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            #nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			#nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             #nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class Cifar_Decoder_1536(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_1536, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             #nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Cifar_Encoder_2048(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_2048, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            #nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            #nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			#nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             #nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class Cifar_Decoder_2048(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_2048, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             #nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


# ===============================================================================================================
def double_convolution(in_channels, out_channels):
    """
    In the original paper implementation, the convolution operations were
    not padded but we are padding them here. This is because, we need the 
    output result size to be same as input size.
    """
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv_op

class UNet_Encoder(nn.Module):
    def __init__(self, num_classes):
        super(UNet_Encoder, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = double_convolution(3, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        # down_9 = self.down_convolution_5(down_8)
        return down_8


class UNet_Decoder(nn.Module):
    def __init__(self, num_classes):
        super(UNet_Decoder, self).__init__()
        # Expanding path.
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512,
            kernel_size=2, 
            stride=2)
        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_convolution_1 = double_convolution(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=2, 
            stride=2)
        self.up_convolution_2 = double_convolution(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2, 
            stride=2)
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2, 
            stride=2)
        self.up_convolution_4 = double_convolution(128, 64)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(
            in_channels=64, out_channels=num_classes, 
            kernel_size=1
        )

    def forward(self, x):
        up_1 = self.up_transpose_1(x)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        out = self.out(x)
        return out

    
if __name__ == '__main__':
    a = torch.randn(4, 3, 512, 1028)
    b = UNet_Encoder(20)(a)
    print(b.shape)