# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
#from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

from .spme_unet import transnext_tiny, transnext_small, transnext_base,TransNeXtDecoder


class SwinUnet(nn.Module):
    def __init__(self, config):
        super(SwinUnet, self).__init__()

        self.config = config


        self.backbone = transnext_base(
            img_size=config.DATA.IMG_SIZE,
            pretrain_size=config.DATA.IMG_SIZE,

            pretrained=config.MODEL.PRETRAIN_CKPT,
        )
        # self.backbone = pvt_v2_b2()
        print('Model %s created, param count: %d' %
              (' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))

        self.decoder = TransNeXtDecoder(

        )
        print('Model %s created, param count: %d' %
              ('decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        #Model  backbone:  created, param count: 63632240
        #Model decoder:  created, param count: 115694587

    def forward(self, x):
            if x.size()[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            features = self.backbone.forward_features(x)
            logits = self.decoder(features)
            return logits

    def load_from(self, config):
        # pretrained_path = '/storage/SA/Swin-Unet-main/Swin-Unet-main/pretrained_ckpt/pvt_v2_b2.pth'
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            print("prekeys",self.backbone.state_dict().keys())
            # if "model"  not in pretrained_dict:
            #     print("---start load pretrained modle by splitting---")
            #     pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
            #     for k in list(pretrained_dict.keys()):
            #         if "output" in k:
            #             print("delete key:{}".format(k))
            #             del pretrained_dict[k]
            self.backbone.load_state_dict(pretrained_dict,strict=False)
            #     # print(msg)
            #     return
            # pretrained_dict = pretrained_dict['model']
            # print("---start load pretrained modle of swin encoder---")
            #
            # model_dict = self.backbone.state_dict()
            # full_dict = copy.deepcopy(pretrained_dict)
            # for k, v in pretrained_dict.items():
            #     if "layers." in k:
            #         current_layer_num = 3-int(k[7:8])
            #         current_k = "layers_up." + str(current_layer_num) + k[8:]
            #         full_dict.update({current_k:v})
            # for k in list(full_dict.keys()):
            #     if k in model_dict:
            #         if full_dict[k].shape != model_dict[k].shape:
            #             print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
            #             del full_dict[k]

            # msg = self.backbone.load_state_dict(full_dict, strict=False)
            # print("msg",msg)
        else:
            print("none pretrain")
 

