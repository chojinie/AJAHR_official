import types
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CBAMChannelAttention(nn.Module):
    """
    CBAM의 Channel Attention 부분
    """
    def __init__(self, in_planes, ratio=16):
        super(CBAMChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class CBAMSpatialAttention(nn.Module):
    """
    CBAM의 Spatial Attention 부분
    """
    def __init__(self, kernel_size=7):
        super(CBAMSpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)

class CBAMBlock(nn.Module):
    """
    CBAM 모듈 전체 (Channel Attention -> Spatial Attention)
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = CBAMChannelAttention(in_planes, ratio=ratio)
        self.sa = CBAMSpatialAttention(kernel_size=kernel_size)
        
    def forward(self, x):
        x_out = x * self.ca(x)
        x_out = x_out * self.sa(x_out)
        return x_out
    
class HeatmapGenerator(nn.Module):
    def __init__(self, image_size, num_keypoints, sigma=2, noise_std=1.5):
        """
        Keypoint Heatmap을 생성하는 모듈 (노이즈 추가 기능 포함)
        Args:
            image_size (tuple): (H, W)
            num_keypoints (int): Keypoint 개수
            sigma (float): Heatmap의 Gaussian 분포 표준편차
            noise_std (float): 최대 노이즈 std (100%에 해당)
        """
        super(HeatmapGenerator, self).__init__()
        self.image_size = image_size
        self.num_keypoints = num_keypoints
        self.sigma = sigma
        self.noise_std = noise_std

        H, W = image_size
        y = torch.arange(0, H).view(H, 1).float()
        x = torch.arange(0, W).view(1, W).float()
        self.register_buffer('x_grid', x)
        self.register_buffer('y_grid', y)

    def forward(self, keypoints, apply_noise=True, noise_ratio=0):
        """
        Args:
            keypoints (torch.Tensor): (B, K, 2) or (B, K, 3)
            apply_noise (bool): 노이즈 적용 여부
            noise_ratio (int): 에너지 기준 노이즈 비율 (0, 25, 50, 75, 100 중 하나)
        Returns:
            heatmaps (torch.Tensor): (B, K, H, W)
        """
        assert noise_ratio in [0, 25, 50, 75, 100], "noise_ratio must be one of [0, 25, 50, 75, 100]"

        if keypoints.size(-1) == 3:
            keypoints = keypoints[..., :2]

        B, K, _ = keypoints.shape
        H, W = self.image_size

        if apply_noise and noise_ratio > 0:
            energy_ratio = noise_ratio / 100.0
            scaled_std = self.noise_std * (energy_ratio ** 0.5)
            noise = torch.randn_like(keypoints) * scaled_std
            keypoints = keypoints + noise
            keypoints = keypoints.clamp(min=0, max=max(H, W))

        x_grid = self.x_grid.unsqueeze(0).unsqueeze(0)
        y_grid = self.y_grid.unsqueeze(0).unsqueeze(0)

        keypoints_x = keypoints[..., 0].unsqueeze(-1).unsqueeze(-1)
        keypoints_y = keypoints[..., 1].unsqueeze(-1).unsqueeze(-1)

        heatmaps = torch.exp(- ((x_grid - keypoints_x)**2 + (y_grid - keypoints_y)**2) / (2 * self.sigma**2))
        return heatmaps

class AmpClassifierHead(nn.Module):
    def __init__(self, num_keypoints=12, sigma=2, image_size=(256,256), pretrained=True):
        super(AmpClassifierHead, self).__init__()
        in_channels = 15  # RGB 이미지 + keypoint heatmap 원래는 3 + num_keypoints
        self.heatmap_generator = HeatmapGenerator(image_size=image_size, num_keypoints=num_keypoints, sigma=sigma, noise_std=1.5,)
        
        self.resnet = models.resnet34(pretrained=pretrained)
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(in_channels,
                                      old_conv.out_channels,
                                      kernel_size=old_conv.kernel_size,
                                      stride=old_conv.stride,
                                      padding=old_conv.padding,
                                      bias=old_conv.bias)
        with torch.no_grad():
            # 기존 가중치는 RGB 채널에 적용
            self.resnet.conv1.weight[:, :3] = old_conv.weight
            # 추가 채널에 대해서는 평균값으로 초기화
            if in_channels > 3:
                avg_weight = old_conv.weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
                self.resnet.conv1.weight[:, 3:] = avg_weight.repeat(1, in_channels-3, 1, 1)
        
        self.feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # 분류 head (원하는 출력 개수에 맞게 수정)
        self.head1 = nn.Linear(self.feature_dim, 4) # 예: 0,1,2,12
        self.head2 = nn.Linear(self.feature_dim, 4) # 예: 3,4,5,12
        self.head3 = nn.Linear(self.feature_dim, 4)
        self.head4 = nn.Linear(self.feature_dim, 4)
        self.projection = nn.Linear(512, 1280)
        
        # --- CBAM 적용: ResNet의 각 BasicBlock에 CBAM 모듈 삽입 ---
        for layer in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            for module in layer:
                # BasicBlock의 conv2 출력 채널 수를 사용합니다.
                module.cbam = CBAMBlock(module.conv2.out_channels)
        
        # BasicBlock의 forward 함수를 오버라이드하여, conv2 후에 CBAM을 적용합니다.
        def forward_with_cbam(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            
            out = self.conv2(out)
            out = self.bn2(out)
            
            # CBAM 적용
            out = self.cbam(out)
            
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out
        
        for layer in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            for module in layer:
                module.forward = types.MethodType(forward_with_cbam, module)
    
    def forward(self, image, keypoints, apply_noise=True):  # TODO: depth 입력 추가 가능
        heatmaps = self.heatmap_generator(keypoints, apply_noise)
        # image: (B, 3, H, W), heatmaps: (B, num_keypoints, H, W)
        #여기부분바꿈0505
        x = torch.cat([image, heatmaps], dim=1)  # (B, 3+num_keypoints, H, W)
        # x = heatmaps
        # 만약 depth 등의 다른 입력 채널을 추가하고 싶으면 아래처럼 사용:
        # x = torch.cat([image, heatmaps, depth], dim=1)
        
        features = self.resnet(x)  # (B, feature_dim)
        to_token = self.projection(features)
        # 각 head의 로짓 계산
        logits1 = self.head1(features)
        logits2 = self.head2(features)
        logits3 = self.head3(features)
        logits4 = self.head4(features)
        
        return {
            'head1': logits1,
            'head2': logits2,
            'head3': logits3,
            'head4': logits4
        }, to_token

