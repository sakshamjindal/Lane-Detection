import os
import torch
import torch.nn as nn
import torch.nn.parallel

from .unet import UNet
from .enet import ENet
from .deeplabv3 import DeepLabv3
from .deeplabv3_plus import DeepLabv3_plus
from .scnn import SCNN
from .erfnet import ERFNet

def get_network(input_channels, num_classes, model_name, kwargs):
    
    if model_name == "enet":
        M = ENet(input_ch=input_channels, output_ch=num_classes)
        if (kwargs["pretrained"]):
            print("Loading pretrained Enet Model")
            M.load_state_dict(torch.load(kwargs["pretrained_model"]))
        else:
            pass
        return M
            
    elif model_name == "deeplabv3":
        if kwargs is None:
            kwargs = {"pretrained": True}
        print("Initializing Deeplabv3...")
        return DeepLabv3(num_classes, kwargs["pretrained"])

    elif model_name == "deeplabv3_plus":
        if kwargs is None:
            kwargs = {"pretrained": True}
        print("Initializing Deeplabv3 Plus...")
        return DeepLabv3_plus(num_classes,kwargs["pretrained_classifier"],kwargs["pretrained"])
    
    elif model_name == "scnn":    
        if kwargs is None:
            kwargs = {"pretrained": True}
        print("Initializing SCNN...")
        
        model = SCNN(input_size = [800, 288],pretrained=kwargs["pretrained"])
        
        if kwargs["pretrained"]:
            model.load_state_dict(torch.load(kwargs["pretrained_model"])["net"])
        else:
            pass
            
        #transfer learning
        model.layer2 = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(128, num_classes, 1)  # get (nB, 5, 36, 100)
            )
        return model
    
    elif model_name == "erfnet" :
        if kwargs is None:
            kwargs = {"pretrained": True}
        print("Initializing ERFNet...")
        
        model = ERFNet(num_classes=37)
        # Remove data parallelism of the state dictionary
        
        if (kwargs["pretrained"]):
            state_dict = torch.load(kwargs["pretrained_model"])["state_dict"]
            new_state_dict = {}
            for k,v in state_dict.items():
                new_state_dict[k[7:]] = v       

            model.load_state_dict(new_state_dict,strict=False)
        else:
            pass
        
        model.decoder.output_conv = (nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True))        
        return model
            
        
#     if(cfg["arch"]["arch_name"]=="unet"):
#         return UNet(num_classes=cfg["data"]["num_classes"])

#     elif(cfg["arch"]["arch_name"]=="enet"):
#         return ENet(num_classes=cfg["data"]["num_classes"])
