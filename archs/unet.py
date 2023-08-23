# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmseg.apis import MMSegInferencer

def build_unet():
    model_name = 'unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024'
    model = UNet(model_name)
    return model

class UNet():
    def __init__(self, model_name) -> None:
        self.inferencer = MMSegInferencer(model=model_name)
        # self.model = init_model(config_path, checkpoint_path, device=device)
    
    def __call__(self, images):
        images = images.permute(0,2,3,1).numpy()
        # segDataSample = inference_model(model=self.model, img=images)
        # result = show_result_pyplot(self.model, images, segDataSample)
        result = self.inferencer(images, show=False)
        result = torch.tensor(result['predictions'])
        return result