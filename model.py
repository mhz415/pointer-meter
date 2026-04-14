import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class H_MLP_Optimized(nn.Module):
    def __init__(self, dim):
        super(H_MLP_Optimized, self).__init__()
        # 💡 核大小为 7，配合 dilation=2，实际感受野达到 13 像素
        self.horizontal = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 6), dilation=(1, 2), groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.vertical = nn.Sequential(
            nn.Conv2d(dim, dim, (7, 1), padding=(6, 0), dilation=(2, 1), groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.horizontal(x)
        v = self.vertical(x)
        return x * self.fuse(h + v)


class CCM_Light(nn.Module):
    def __init__(self, dim):
        super(CCM_Light, self).__init__()
        mid_dim = dim // 4
        self.conv_path = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1, groups=mid_dim, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.gate = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x_feat = self.conv_path(x)
        x = res + x_feat * self.gate(x_feat)
        return self.relu(x)


class SDM_2D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SDM_2D, self).__init__()
        self.diff_conv = nn.Sequential(
            nn.Conv2d(in_channels + num_classes, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )

    def forward(self, feat, pre_mask):
        diff = self.diff_conv(torch.cat([feat, pre_mask], dim=1))
        return pre_mask + diff


class CLIFF(nn.Module):
    def __init__(self, high_channels, low_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(CLIFF, self).__init__()
        self.high_transform = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, 1, bias=False),
            BatchNorm(low_channels),
            nn.ReLU(inplace=True)
        )
        self.align_conv = nn.Sequential(
            nn.Conv2d(low_channels, low_channels, 3, padding=1, groups=low_channels, bias=False),
            BatchNorm(low_channels)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(low_channels * 3, out_channels, 3, padding=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, high_feat, low_feat):
        high_feat_up = F.interpolate(self.high_transform(high_feat), size=low_feat.size()[2:], mode='bilinear',
                                     align_corners=True)
        high_feat_up = self.align_conv(high_feat_up)
        enhanced_low_feat = low_feat * self.sigmoid(high_feat_up)
        return self.fusion_conv(torch.cat([high_feat_up, low_feat, enhanced_low_feat], dim=1))

# =========================================================================
#  ASPP 模块 (适配多尺度特征)
# =========================================================================
class ASPP(nn.Module):
    def __init__(self, output_stride, BatchNorm, in_channels):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        else:
            dilations = [1, 12, 24, 36]

        self.aspp1 = nn.Sequential(nn.Conv2d(in_channels, 256, 1, bias=False), BatchNorm(256), nn.ReLU())
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=dilations[1], dilation=dilations[1], bias=False), BatchNorm(256),
            nn.ReLU())
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=dilations[2], dilation=dilations[2], bias=False), BatchNorm(256),
            nn.ReLU())
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=dilations[3], dilation=dilations[3], bias=False), BatchNorm(256),
            nn.ReLU())
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, 256, 1, bias=False),
                                             BatchNorm(256), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(1280, 256, 1, bias=False), BatchNorm(256), nn.ReLU())

    def forward(self, x):
        x1, x2, x3, x4 = self.aspp1(x), self.aspp2(x), self.aspp3(x), self.aspp4(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=x4.size()[2:], mode='bilinear', align_corners=True)
        return self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))

# =========================================================================
# 6主模型 DeepLab (MSH-PPNet Optimized)
# =========================================================================
class DeepLab(nn.Module):
    def __init__(self, num_classes=3, backbone='mobilenetv4', output_stride=16):
        super(DeepLab, self).__init__()
        # 使用 timm 加载骨干网络
        self.backbone = timm.create_model('mobilenetv4_conv_medium', pretrained=True, features_only=True)
        chans = self.backbone.feature_info.channels()

        self.aspp = ASPP(output_stride, nn.BatchNorm2d, in_channels=chans[3])

        # 浅层特征处理 (采用深度可分离卷积提速)
        self.low_conv = nn.Sequential(
            nn.Conv2d(chans[1], 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cliff = CLIFF(256, 48, 256)
        self.ccm = CCM_Light(dim=256)
        self.h_mlp = H_MLP_Optimized(256)
        self.classifier = nn.Conv2d(256, num_classes, 1)
        self.sdm = SDM_2D(in_channels=256, num_classes=num_classes)

    def forward(self, input):
        feats = self.backbone(input)

        # Encoder 提取
        x = self.aspp(feats[3])

        # 解码融合
        low_feat = self.low_conv(feats[1])
        x = self.cliff(x, low_feat)

        # 创新模块组合
        x = self.ccm(x)
        x = self.h_mlp(x)

        # 边界修正
        pre_output = self.classifier(x)
        final_output = self.sdm(x, pre_output)

        # 上采样回原图大小
        return F.interpolate(final_output, size=input.size()[2:], mode='bilinear', align_corners=True)