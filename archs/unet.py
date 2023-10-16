# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmseg.apis import MMSegInferencer
import torch.nn.functional as F

def build_unet(device):
    model_name = 'unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024'
    model = UNet(model_name, device)
    return model

class UNet():
    def __init__(self, model_name, device) -> None:
        self.inferencer = MMSegInferencer(model=model_name, device=device)
        # self.model = init_model(config_path, checkpoint_path, device=device)
    
    def __call__(self, images):
        images = list(images.permute(0,2,3,1).detach().cpu().numpy())
        # segDataSample = inference_model(model=self.model, img=images)
        # result = show_result_pyplot(self.model, images, segDataSample)
        result = self.inferencer(images, show=False)
        result = torch.tensor(result['predictions'])
        if len(result.shape) == 2:
            result = result.unsqueeze(0)
        result = F.one_hot(result, num_classes=20).permute(0,3,1,2).float()
        return result
    
class DeepLab3():
    def __init__(self, model_name, device) -> None:
        self.model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
        self.model.eval()

    def __call__(self, images):
        images = list(images.permute(0,2,3,1).detach().cpu().numpy())
        output = self.model(images)['out']
        return output