# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmseg.apis import MMSegInferencer

def build_unet(device):
    model_name = 'deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024'
    model = UNet(model_name, device)
    return model

class UNet():
    def __init__(self, model_name, device) -> None:
        self.inferencer = MMSegInferencer(model=model_name).to(device)
        # self.model = init_model(config_path, checkpoint_path, device=device)
    
    def __call__(self, images):
        # segDataSample = inference_model(model=self.model, img=images)
        # result = show_result_pyplot(self.model, images, segDataSample)
        result = self.inferencer(images, show=True, wait_time=0.5)
        result = torch.tensor(result['predictions'])
        return result