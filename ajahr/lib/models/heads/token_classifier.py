import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.join(__file__.replace(os.path.basename(__file__), ''), '..', '..', '..', '..'))

from .utils import (constant_init, normal_init)
from .modules import MixerLayer, FCBlock, BasicBlock
from tokenization.models.vanilla_pose_vqvae import DecodeTokens as AmpDecodeTokens
from tokenization.models.vanilla_pose_vqvae import DecodeTokens as NormalDecodeTokens

class ProxyAmp(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.set_gpu = False
    def tokenize(self, x):
        if not self.set_gpu:
            self.tokenizer = self.tokenizer.to(x.device)
            self.set_gpu = True
        return self.tokenizer(x)

class ProxyNoraml(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.set_gpu = False
    def tokenize(self, x):
        if not self.set_gpu:
            self.tokenizer = self.tokenizer.to(x.device)
            self.set_gpu = True
        return self.tokenizer(x)

class TokenClassfier(nn.Module):
    def __init__(self, in_channels=2048, token_num=40, token_class_num=2046, token_code_dim=None, 
                 tokenizer_checkpoint_path_amp=None, tokenizer_checkpoint_path_normal=None, tokenizer_type='Vanilla'):
        super().__init__()

        self.conv_num_blocks = 1
        self.dilation = 1
        self.conv_channels = 256
        self.hidden_dim = 64
        self.num_blocks = 4
        self.hidden_inter_dim = 256
        self.token_inter_dim = 64
        self.dropout = 0.0

        self.token_num = token_num           
        self.token_class_num = token_class_num 

        self.token_code_dim = token_code_dim

        self.mixer_trans = FCBlock(
            in_channels, 
            self.token_num * self.hidden_dim)

        self.mixer_head = nn.ModuleList(
            [MixerLayer(self.hidden_dim, self.hidden_inter_dim,
                        self.token_num, self.token_inter_dim,  
                        self.dropout) for _ in range(self.num_blocks)])
        self.mixer_norm_layer = FCBlock(
            self.hidden_dim, self.hidden_dim)

        self.class_pred_layer = nn.Linear(self.hidden_dim, self.token_class_num)

        tokenizer_type_amp = 'amp'
        tokenizer_type_normal = 'normal'
        
        self.tokenizer_amp = eval(f'{tokenizer_type_amp.capitalize()}DecodeTokens')(tokenizer_checkpoint_path_amp)
        self.tokenizer_normal = eval(f'{tokenizer_type_normal.capitalize()}DecodeTokens')(tokenizer_checkpoint_path_normal)
        self.tokenize_amp = ProxyAmp(self.tokenizer_amp).tokenize
        self.tokenize_normal = ProxyNoraml(self.tokenizer_normal).tokenize

        self.register_buffer('zero_tensor', torch.tensor([0, 0, 0, 0]))

    
    def forward(self, x, mask):
        """
        input x shape : [batch, 1024]
        """
        batch_size = x.shape[0]
        cls_feat = self.mixer_trans(x) 
        cls_feat = cls_feat.reshape(batch_size, self.token_num, -1)  

        for mixer_layer in self.mixer_head:
            cls_feat = mixer_layer(cls_feat)
        cls_feat = self.mixer_norm_layer(cls_feat)  

        cls_logits = self.class_pred_layer(cls_feat)  # [Batch, 160, 2048]

        cls_logits_softmax = cls_logits.softmax(-1)  # [Batch, 160, 2048]

        mask = (mask == torch.tensor([0, 0, 0, 0], device=x.device)).all(dim=1)
        
        smpl_thetas6D_normal = (
            self.tokenize_normal(cls_logits_softmax[mask]).to(x.dtype) if (mask).any() else None
            )
        smpl_thetas6D_amp = (
            self.tokenize_amp(cls_logits_softmax[~mask]).to(x.dtype) if (~mask).any() else None
            )

        smpl_thetas6D = torch.zeros(batch_size, 21 * 6, dtype=x.dtype, device=x.device)


        if smpl_thetas6D_normal is not None:
            smpl_thetas6D[mask] = smpl_thetas6D_normal.reshape(-1, 21*6)
        if smpl_thetas6D_amp is not None:
            smpl_thetas6D[~mask] = smpl_thetas6D_amp.reshape(-1, 21*6)


        return smpl_thetas6D, cls_logits_softmax


    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_cls_head(self, conv_channels, conv_num_blocks, dilation):
        feature_convs = []
        feature_conv = self._make_layer(
            BasicBlock,
            conv_channels,
            conv_channels,
            conv_num_blocks,
            dilation=dilation)
        feature_convs.append(feature_conv)
        
        return nn.ModuleList(feature_convs)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    # ---------- 개선 3: frozen_tokenizer 메서드 수정 ----------
    def frozen_tokenizer(self):
        """
        두 토크나이저(amp, normal) 모두 evaluation 모드로 전환하고,
        모든 파라미터에 대해 gradient 계산을 비활성화합니다.
        """
        self.tokenizer_amp.eval()
        self.tokenizer_normal.eval()
        for param in self.tokenizer_amp.parameters():
            param.requires_grad = False
        for param in self.tokenizer_normal.parameters():
            param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
