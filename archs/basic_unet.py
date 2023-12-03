import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transform


class Convblock(nn.Module):
    def __init__(self,input_channel,output_channel,kernal=3,stride=1,padding=1):
        super(Convblock,self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernal,stride,padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel,output_channel,kernal),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.convblock(x)
        return x


class UnetEncoder(nn.Module):
    
    def __init__(self,input_channel):
        super(UnetEncoder,self).__init__()
        self.conv1 = Convblock(input_channel,32)
        self.conv2 = Convblock(32,64)
        self.conv3 = Convblock(64,128)
        self.conv4 = Convblock(128,256)
        self.conv5 = Convblock(256,512)
        self.conv6 = Convblock(512,1024)
        self.neck = nn.Conv2d(1024,2048,3,1)
        
    def forward(self,x, decoder=None):
        
        # Encoder Network
        
        # Conv down 1
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1,kernel_size=2,stride=2)
        # Conv down 2
        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2,kernel_size=2,stride=2)
        # Conv down 3
        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3,kernel_size=2,stride=2)
        # Conv down 4
        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4,kernel_size=2,stride=2)
        # Conv down 5
        conv5 = self.conv5(pool4)
        pool5 = F.max_pool2d(conv5,kernel_size=2,stride=2)
        # Conv down 6
        conv6 = self.conv6(pool5)
        pool6 = F.max_pool2d(conv6,kernel_size=2,stride=1)
        
        # BottelNeck
        neck = self.neck(pool6)
        if decoder is not None:
            decoder.set_conv([conv6.detach(), conv5.detach(), conv4.detach(), \
                conv3.detach(), conv2.detach(), conv1.detach()])
        
        return neck
    

class UnetDecoder(nn.Module):
    def __init__(self,retain_size=None):
        super(UnetDecoder,self).__init__()
        self.upconv6 = nn.ConvTranspose2d(2048,1024,3,2,0,1)
        self.dconv6 = Convblock(2048, 1024)
        self.upconv5 = nn.ConvTranspose2d(1024,512,3,2,0,1)
        self.dconv5 = Convblock(1024, 512)
        self.upconv4 = nn.ConvTranspose2d(512,256,3,2,0,1)
        self.dconv4 = Convblock(512,256)
        self.upconv3 = nn.ConvTranspose2d(256,128,3,2,0,1)
        self.dconv3 = Convblock(256,128)
        self.upconv2 = nn.ConvTranspose2d(128,64,3,2,0,1)
        self.dconv2 = Convblock(128,64)
        self.upconv1 = nn.ConvTranspose2d(64,32,3,2,0,1)
        self.dconv1 = Convblock(64,32)
        self.out = nn.Conv2d(32,3,1,1)
        self.retain_size = retain_size
        
    def forward(self,x):
        conv6, conv5, conv4, conv3, conv2, conv1 = self.conv
        
        # Decoder Network
        
        upconv6 = self.upconv6(x)
        croped = self.crop(conv6,upconv6)
        dconv6 = self.dconv6(torch.cat([upconv6,croped],1))
        # Upconv 1
        upconv5 = self.upconv5(dconv6)
        croped = self.crop(conv5,upconv5)
        dconv5 = self.dconv5(torch.cat([upconv5,croped],1))
        
        upconv4 = self.upconv4(dconv5)
        croped = self.crop(conv4,upconv4)
        # Making the skip connection
        dconv4 = self.dconv4(torch.cat([upconv4,croped],1))
        # Upconv 2
        upconv3 = self.upconv3(dconv4)
        croped = self.crop(conv3,upconv3)
        # Making the skip connection 2
        dconv3 = self.dconv3(torch.cat([upconv3,croped],1))
        # Upconv 3
        upconv2 = self.upconv2(dconv3)
        croped = self.crop(conv2,upconv2)
        # Making the skip connection 3
        dconv2 = self.dconv2(torch.cat([upconv2,croped],1))
        # Upconv 4
        upconv1 = self.upconv1(dconv2)
        croped = self.crop(conv1,upconv1)
        # Making the skip connection 4
        dconv1 = self.dconv1(torch.cat([upconv1,croped],1))
        # Output Layer
        out = self.out(dconv1)
        
        if self.retain_size is not None:
            out = F.interpolate(out,list(self.retain_size)[2:])

        return out
    
    def crop(self,input_tensor,target_tensor):
        # For making the size of the encoder conv layer and the decoder Conv layer same
        _,_,H,W = target_tensor.shape
        return transform.CenterCrop([H,W])(input_tensor)
    
    def set_conv(self, conv):
        self.conv = conv


class Unet_2Block(nn.Module):
    def __init__(self,input_channel, retain_size=None):
        self.encoder = UnetEncoder_2Block(input_channel)
        self.decoder = UnetDecoder_2Block(retain_size)
    
    def forward(self,x):
        out = self.encoder(x, self.decoder)
        out = self.decoder(out)
        return out


class UnetEncoder_2Block(nn.Module):
    def __init__(self,input_channel):
        super(UnetEncoder,self).__init__()
        self.conv1 = Convblock(input_channel,32,padding=0)
        self.conv2 = Convblock(32,64,padding=0)
        self.neck = Convblock(64,128,padding=0)

    def forward(self,x, decoder=None):

        # Encoder Network

        # Conv down 1
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1,kernel_size=2)
        print(pool1.shape)
        # Conv down 2
        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2,kernel_size=2)
        print(pool2.shape)
        # BottelNeck
        neck = self.neck(pool2)
        if decoder is not None:
            decoder.set_conv([conv2.detach(), conv1.detach()])

        return neck
    

class UnetDecoder_2Block(nn.Module):
    def __init__(self,retain_size=None):
        super(UnetDecoder,self).__init__()
        self.upconv2 = nn.ConvTranspose2d(128,64,3,2,0,1)
        self.dconv2 = Convblock(128,64)
        self.upconv1 = nn.ConvTranspose2d(64,32,3,2,0,1)
        self.dconv1 = Convblock(64,32)
        self.out = nn.Conv2d(32,3,1,1)
        self.retain_size = retain_size

    def forward(self,x):
        conv2, conv1 = self.conv

        # Decoder Network
        # Upconv 2
        upconv2 = self.upconv2(x)
        croped = self.crop(conv2,upconv2)
        # Making the skip connection 3
        dconv2 = self.dconv2(torch.cat([upconv2,croped],1))
        # Upconv 1
        upconv1 = self.upconv1(dconv2)
        croped = self.crop(conv1,upconv1)
        # Making the skip connection 4
        dconv1 = self.dconv1(torch.cat([upconv1,croped],1))
        # Output Layer
        out = self.out(dconv1)

        if self.retain_size is not None:
            out = F.interpolate(out,list(self.retain_size)[2:])

        return out

    def crop(self,input_tensor,target_tensor):
        # For making the size of the encoder conv layer and the decoder Conv layer same
        _,_,H,W = target_tensor.shape
        return transform.CenterCrop([H,W])(input_tensor)

    def set_conv(self, conv):
        self.conv = conv