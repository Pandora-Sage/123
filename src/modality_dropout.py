"""
Modality Dropout Strategy V2.2
===============================
折中策略: Text 低概率丢弃

训练时: A/V 高概率, L 低概率（让模型学会处理但不影响主性能）
测试时: 支持全部 7 种缺失模式
"""

import torch
import torch.nn as nn


class ModalityDropout(nn.Module):

    def __init__(self, miss_prob=0.5, has_video=True):
        super().__init__()
        self.miss_prob = miss_prob
        self.has_video = has_video

        if has_video:
            # Text dropout 占缺失样本的 25%，足以训练相关模块
            self.patterns = ['A', 'V', 'AV', 'L', 'LV', 'LA']
            self.weights =  [0.28, 0.28, 0.19, 0.08, 0.10, 0.07]
        else:
            self.patterns = ['A', 'L']
            self.weights = [0.85, 0.15]

        # 缺失类型映射
        self.pattern_to_type = {
            '': 0, 'A': 1, 'V': 2, 'AV': 3,
            'L': 4, 'LV': 5, 'LA': 6,
        }

    def forward(self, batch_size, device, phase='train'):
        flags = {
            'L': torch.zeros(batch_size, device=device),
            'A': torch.zeros(batch_size, device=device),
            'V': torch.zeros(batch_size, device=device),
        }
        miss_type = torch.zeros(batch_size, dtype=torch.long, device=device)

        if phase != 'train' or not self.training:
            return flags, miss_type

        miss_mask = torch.rand(batch_size, device=device) < self.miss_prob
        if not miss_mask.any():
            return flags, miss_type

        n_miss = miss_mask.sum().item()
        weights_tensor = torch.tensor(self.weights, device=device, dtype=torch.float)
        pattern_indices = torch.multinomial(weights_tensor, int(n_miss), replacement=True)
        miss_indices = torch.where(miss_mask)[0]

        for i, idx in enumerate(miss_indices):
            pattern = self.patterns[pattern_indices[i].item()]
            for mod in pattern:
                flags[mod][idx] = 1.0
            miss_type[idx] = self.pattern_to_type[pattern]

        return flags, miss_type

    def set_test_missing(self, batch_size, device, missing_modalities=''):
        """测试时支持全部 7 种缺失模式"""
        flags = {
            'L': torch.zeros(batch_size, device=device),
            'A': torch.zeros(batch_size, device=device),
            'V': torch.zeros(batch_size, device=device),
        }
        for mod in missing_modalities:
            if mod in flags:
                flags[mod] = torch.ones(batch_size, device=device)

        type_id = self.pattern_to_type.get(missing_modalities, 0)
        miss_type = torch.full((batch_size,), type_id, dtype=torch.long, device=device)
        return flags, miss_type