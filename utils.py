import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SegmentationLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # 边缘提取核 (用于辅助检测)
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('edge_kernel', kernel)

    def dice_loss(self, predict, target):
        predict = F.softmax(predict, dim=1)
        valid_mask = (target != self.ignore_index).float().unsqueeze(1)
        target_onehot = F.one_hot(target.clamp(min=0), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        target_onehot = target_onehot * valid_mask
        predict = predict * valid_mask
        intersection = torch.sum(predict * target_onehot, dim=(2, 3))
        union = torch.sum(predict, dim=(2, 3)) + torch.sum(target_onehot, dim=(2, 3))
        return 1.0 - torch.mean((2.0 * intersection + 1e-5) / (union + 1e-5))

    def boundary_loss(self, predict, target):
        """利用 SDM 思想增强的边缘损失"""
        predict_softmax = F.softmax(predict, dim=1)
        target_onehot = F.one_hot(target.clamp(min=0), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        def get_edges(x):
            channels = x.shape[1]
            current_kernel = self.edge_kernel.expand(channels, 1, 3, 3).to(x.device)
            return torch.abs(F.conv2d(x, current_kernel, padding=1, groups=channels))

        return F.mse_loss(get_edges(predict_softmax), get_edges(target_onehot))

    def forward(self, predict, target):

        ce_loss = F.cross_entropy(predict, target, ignore_index=self.ignore_index)

        d_loss = self.dice_loss(predict, target)

        b_loss = self.boundary_loss(predict, target)

        # 建议权重配比
        return ce_loss + 1.0 * d_loss + 0.1 * b_loss


# ==============================================================================
# 2. 评估器 (保持不变)
# ==============================================================================
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return np.nanmean(Acc)

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return np.nanmean(MIoU)

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return (freq[freq > 0] * iu[freq > 0]).sum()

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        return count.reshape(self.num_class, self.num_class)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
