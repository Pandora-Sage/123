"""
Modality-Aware Prompt V3 (简洁版)
==================================
硬切换：真实模态用 real_prompt，生成模态用 gen_prompt
不做动态混合，更简洁，更不容易过拟合

vs MPLMM (ACL 2024):
  MPLMM: present/absent 二选一（和我们类似）
  Ours:  real/gen 二选一 + Missing-Type 全局上下文
         real/gen prompt 是多个可学习 token（不是单个 embedding）
"""

import torch
import torch.nn as nn


class ModalityAwarePrompt(nn.Module):

    def __init__(self, embed_dim, prompt_len=4, modalities=('L', 'A', 'V')):
        super().__init__()
        self.embed_dim = embed_dim
        self.prompt_len = prompt_len

        # 每个模态两套 prompt：真实版 / 生成版
        self.real_prompts = nn.ParameterDict()
        self.gen_prompts = nn.ParameterDict()
        for mod in modalities:
            self.real_prompts[mod] = nn.Parameter(torch.randn(prompt_len, embed_dim) * 0.02)
            self.gen_prompts[mod] = nn.Parameter(torch.randn(prompt_len, embed_dim) * 0.02)

        # Missing-Type Prompt: 7种缺失模式的全局上下文
        self.type_prompt = nn.Parameter(torch.randn(7, prompt_len, embed_dim) * 0.02)

    def get_signal_prompt(self, modality, flag, batch_size):
        """根据 flag 硬切换选择 real 或 gen prompt"""
        real_p = self.real_prompts[modality].unsqueeze(1).expand(-1, batch_size, -1)
        gen_p = self.gen_prompts[modality].unsqueeze(1).expand(-1, batch_size, -1)

        mask = flag.unsqueeze(0).unsqueeze(-1)  # (1, batch, 1)
        prompt = (1 - mask) * real_p + mask * gen_p
        return prompt

    def get_type_prompt(self, miss_type):
        """根据缺失模式选择全局上下文"""
        type_p = self.type_prompt[miss_type]  # (batch, prompt_len, dim)
        return type_p.permute(1, 0, 2)        # (prompt_len, batch, dim)

    def forward(self, h_mod, modality, flag, miss_type):
        """
        Args:
            h_mod: (seq, batch, dim) - 模态特征
            modality: 'L' / 'A' / 'V'
            flag: (batch,) - 0=real, 1=generated
            miss_type: (batch,) - 缺失类型 0-6
        Returns:
            prompted: (signal_len + type_len + seq, batch, dim)
        """
        batch_size = h_mod.size(1)
        signal_prompt = self.get_signal_prompt(modality, flag, batch_size)
        type_prompt = self.get_type_prompt(miss_type)

        prompted = torch.cat([signal_prompt, type_prompt, h_mod], dim=0)
        return prompted, signal_prompt