"""
CFMModel V2.2 — Missing-Aware Dynamic Prompt
=============================================
在 V2.1 基础上改进 prompt:
  1. 每个 prompt 接收 (模态summary + 缺失上下文)
  2. prompt_len 增加到 4（可配置）
  3. 所有模态 prompt 统一设计

其他部分完全不动:
  BUG A 修复 + valid_loader 修复 + weight_decay + 低概率 Text dropout

返回: (output, last_hs, gen_loss, ortho_loss, aux_loss)
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
from modules.unimf import MultimodalTransformerEncoder
from modules.position_embedding import SinusoidalPositionalEmbedding
from src.generator import ModalityGenerator, NoiseGate, orthogonality_loss
from src.modality_dropout import ModalityDropout


class BertTextEncoder(nn.Module):
    def __init__(self, language='en', use_finetune=False):
        super().__init__()
        assert language in ['en', 'cn']
        from transformers import BertTokenizer, BertModel
        import os

        possible_paths = [
            f'pretrained_berts/bert_{language}',
            f'pretrained_bert/bert_{language}',
            f'pre_trained_models/bert_{language}',
        ]
        loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                kw = {'do_lower_case': True} if language == 'en' else {}
                self.tokenizer = BertTokenizer.from_pretrained(path, **kw)
                self.model = BertModel.from_pretrained(path)
                loaded = True
                print(f"  - Loaded BERT from: {path}")
                break
        if not loaded:
            model_name = 'bert-base-uncased' if language == 'en' else 'bert-base-chinese'
            print(f"  - Loading BERT from HuggingFace: {model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
        self.use_finetune = use_finetune

    def forward(self, text):
        input_ids = text[:, 0, :].long()
        input_mask = text[:, 1, :].float()
        segment_ids = text[:, 2, :].long()
        if self.use_finetune:
            return self.model(input_ids=input_ids, attention_mask=input_mask,
                              token_type_ids=segment_ids)[0]
        else:
            with torch.no_grad():
                return self.model(input_ids=input_ids, attention_mask=input_mask,
                                  token_type_ids=segment_ids)[0]


class CFMModel(nn.Module):

    PATTERN_DEFS = {
        1: (['A'], ['L', 'V']),
        2: (['V'], ['L', 'A']),
        3: (['A', 'V'], ['L']),
        4: (['L'], ['A', 'V']),
        5: (['L', 'V'], ['A']),
        6: (['L', 'A'], ['V']),
    }

    def __init__(self, hyp_params):
        super().__init__()

        self.modalities = hyp_params.modalities
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.dataset = hyp_params.dataset
        self.use_bert = hyp_params.use_bert
        self.distribute = hyp_params.distribute
        output_dim = hyp_params.output_dim

        self.is_meld = self.dataset in ['meld_senti', 'meld_emo']
        if self.is_meld:
            self.orig_l_len, self.orig_a_len = hyp_params.l_len, hyp_params.a_len
            self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
            self.orig_v_len, self.orig_d_v = 0, 0
        else:
            self.orig_l_len, self.orig_a_len, self.orig_v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
            self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v

        self.l_kernel_size = hyp_params.l_kernel_size
        self.a_kernel_size = hyp_params.a_kernel_size
        self.v_kernel_size = hyp_params.v_kernel_size if not self.is_meld else 1
        self.l_len = self.orig_l_len - self.l_kernel_size + 1
        self.a_len = self.orig_a_len - self.a_kernel_size + 1
        self.v_len = (self.orig_v_len - self.v_kernel_size + 1) if not self.is_meld else 0
        self.has_video = not self.is_meld

        # ==================== BERT ====================
        if self.use_bert:
            self.text_model = BertTextEncoder(
                language=getattr(hyp_params, "language", "en"), use_finetune=True)

        # ==================== 卷积投影 ====================
        self.proj_l = nn.Conv1d(self.orig_d_l, self.embed_dim, kernel_size=self.l_kernel_size)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.embed_dim, kernel_size=self.a_kernel_size)
        if self.has_video:
            self.proj_v = nn.Conv1d(self.orig_d_v, self.embed_dim, kernel_size=self.v_kernel_size)

        # ==================== 位置编码 ====================
        self.pos_enc = SinusoidalPositionalEmbedding(self.embed_dim)

        # ==================== 单模态编码器 ====================
        self.unimodal_encoder_l = self._make_unimodal_encoder()
        self.unimodal_encoder_a = self._make_unimodal_encoder()
        if self.has_video:
            self.unimodal_encoder_v = self._make_unimodal_encoder()
        self.ln_l = nn.LayerNorm(self.embed_dim)
        self.ln_a = nn.LayerNorm(self.embed_dim)
        if self.has_video:
            self.ln_v = nn.LayerNorm(self.embed_dim)

        # ==================== NoiseGate ====================
        self.noise_gate_a = NoiseGate(self.embed_dim)
        if self.has_video:
            self.noise_gate_v = NoiseGate(self.embed_dim)

        # ==================== Generator ====================
        target_dims = {'L': self.orig_d_l, 'A': self.orig_d_a}
        if self.has_video:
            target_dims['V'] = self.orig_d_v
        if self.use_bert:
            target_dims['L'] = 768

        self.generator = ModalityGenerator(
            embed_dim=self.embed_dim,
            target_dims=target_dims,
            num_heads=min(self.num_heads, max(1, self.embed_dim // 8)),
            num_layers=2, num_style_tokens=4,
            dropout=self.attn_dropout
        )

        # ==================== ModalityDropout ====================
        miss_prob = getattr(hyp_params, 'miss_prob', 0.4)
        self.modality_dropout = ModalityDropout(miss_prob=miss_prob, has_video=self.has_video)

        # ==================== Reliability + Missing-Type ====================
        self.reliability_emb = nn.Embedding(2, self.embed_dim)
        self.missing_type_emb = nn.Embedding(7, self.embed_dim)
        self.mt_scale = nn.Parameter(torch.tensor(0.1))

        # ==================== 辅助分类头 ====================
        self.aux_classifier_l = nn.Linear(self.embed_dim, output_dim)
        self.aux_classifier_a = nn.Linear(self.embed_dim, output_dim)
        if self.has_video:
            self.aux_classifier_v = nn.Linear(self.embed_dim, output_dim)

        # ==================== 不确定性头 + Default Tokens ====================
        self.use_gate = getattr(hyp_params, 'use_gate', 1)
        # Per-modality uncertainty heads（各模态生成质量不同，需独立评估）
        def _make_uncertainty_head():
            return nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.ReLU(),
                nn.Linear(self.embed_dim // 2, 1),
                nn.Sigmoid()
            )
        self.uncertainty_head_l = _make_uncertainty_head()
        self.uncertainty_head_a = _make_uncertainty_head()
        if self.has_video:
            self.uncertainty_head_v = _make_uncertainty_head()
        # Adaptive Gate: 生成质量差时用 default token 代替噪声
        if self.use_gate:
            self.default_token_l = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.default_token_a = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.normal_(self.default_token_l, std=0.02)
            nn.init.normal_(self.default_token_a, std=0.02)
            if self.has_video:
                self.default_token_v = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                nn.init.normal_(self.default_token_v, std=0.02)

        # ==================== CLS ====================
        self.cls_len = 33 if self.is_meld else 1
        self.cls = nn.Parameter(torch.zeros(self.cls_len, self.embed_dim))
        nn.init.normal_(self.cls, std=0.02)

        # ==================== Missing-Aware Dynamic Prompt [改进] ====================
        # 输入: cat[模态summary, 缺失上下文] = (batch, embed_dim * 2)
        # 输出: (batch, num_prompts * embed_dim)
        self.num_prompts = getattr(hyp_params, 'prompt_len', 4)

        prompt_in_dim = self.embed_dim * 2  # summary + miss_context
        self.prompt_gen_l = nn.Sequential(
            nn.Linear(prompt_in_dim, self.embed_dim), nn.GELU(),
            nn.Linear(self.embed_dim, self.num_prompts * self.embed_dim))
        self.prompt_gen_a = nn.Sequential(
            nn.Linear(prompt_in_dim, self.embed_dim), nn.GELU(),
            nn.Linear(self.embed_dim, self.num_prompts * self.embed_dim))
        if self.has_video:
            self.prompt_gen_v = nn.Sequential(
                nn.Linear(prompt_in_dim, self.embed_dim), nn.GELU(),
                nn.Linear(self.embed_dim, self.num_prompts * self.embed_dim))

        # ==================== 融合 ====================
        self.fusion_modal_type_embeddings = nn.Embedding(4, self.embed_dim)
        self.fusion_encoder = MultimodalTransformerEncoder(
            embed_dim=self.embed_dim, num_heads=self.num_heads,
            layers=hyp_params.multimodal_layers,
            lens=(self.cls_len, self.l_len, self.a_len),
            modalities=self.modalities,
            attn_dropout=self.attn_dropout, embed_positions=None)

        # ==================== 输出 ====================
        self.att_pool_linear = nn.Linear(self.embed_dim, 1)
        self.proj1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_layer = nn.Linear(self.embed_dim, output_dim)

        self._init_weights()

    def _make_unimodal_encoder(self):
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 2, dropout=self.attn_dropout,
            norm_first=True, batch_first=False)
        return nn.TransformerEncoder(layer, num_layers=2)

    def _init_weights(self):
        skip = ['text_model', 'generator.content_encoder', 'generator.style_provider',
                'generator.noise_gate']
        for name, m in self.named_modules():
            if any(name.startswith(s) for s in skip):
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=self.embed_dim ** -0.5)

    def _encode_modality(self, x, proj, encoder, ln, seq_len, batch_size, device,
                         noise_gate=None):
        x_t = F.dropout(x.transpose(1, 2), p=self.embed_dropout, training=self.training)
        proj_x = proj(x_t).permute(2, 0, 1) * math.sqrt(self.embed_dim)
        actual_seq = proj_x.size(0)
        actual_batch = proj_x.size(1)
        pos_ids = torch.arange(actual_seq, device=device).unsqueeze(0).expand(actual_batch, -1)
        pe = self.pos_enc(pos_ids)
        proj_x = proj_x + pe.transpose(0, 1)
        if noise_gate is not None:
            proj_x = noise_gate(proj_x)
        h = encoder(proj_x)
        h = ln(h)
        return h

    def _make_prompt(self, prompt_gen, summary, miss_ctx, batch_size):
        """生成 missing-aware dynamic prompt"""
        prompt_input = torch.cat([summary, miss_ctx], dim=-1)  # (batch, embed_dim*2)
        prompt = prompt_gen(prompt_input)  # (batch, num_prompts * embed_dim)
        prompt = prompt.view(batch_size, self.num_prompts, self.embed_dim)
        return prompt.permute(1, 0, 2)  # (num_prompts, batch, embed_dim)

    def forward(self, x_l, x_a, x_v=None, phase='train', missing_flags=None,
                miss_type=None, labels=None, mask=None):
        batch_size = x_l.size(0)
        device = x_l.device

        # ==================== 1. BERT ====================
        if self.use_bert:
            x_l_encoded = self.text_model(x_l)
        else:
            x_l_encoded = x_l

        # ==================== 2. 缺失决策 ====================
        if missing_flags is not None:
            flags = missing_flags
            m_type = miss_type if miss_type is not None else torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            flags, m_type = self.modality_dropout(batch_size, device, phase)

        # ==================== 3. 编码 ====================
        h_l = self._encode_modality(x_l_encoded, self.proj_l, self.unimodal_encoder_l,
                                     self.ln_l, self.l_len, batch_size, device)
        h_a = self._encode_modality(x_a, self.proj_a, self.unimodal_encoder_a,
                                     self.ln_a, self.a_len, batch_size, device,
                                     noise_gate=self.noise_gate_a)
        h_v = None
        if self.has_video and x_v is not None:
            h_v = self._encode_modality(x_v, self.proj_v, self.unimodal_encoder_v,
                                         self.ln_v, self.v_len, batch_size, device,
                                         noise_gate=self.noise_gate_v)

        # ==================== 4. 辅助分类损失 ====================
        aux_loss = torch.tensor(0.0, device=device)
        if labels is not None and phase == 'train':
            if self.is_meld:
                # MELD: h_l=(seq_l, batch, dim), h_a=(seq_a, batch, dim), labels=(batch, 33)
                criterion_aux = nn.CrossEntropyLoss()
                seq_l = h_l.size(0)
                seq_a = h_a.size(0)
                aux_pred_l = self.aux_classifier_l(h_l.permute(1, 0, 2).reshape(-1, self.embed_dim))
                aux_pred_a = self.aux_classifier_a(h_a.permute(1, 0, 2).reshape(-1, self.embed_dim))
                labels_l = labels[:, :seq_l].reshape(-1)
                labels_a = labels[:, :seq_a].reshape(-1)
                # 用 mask 过滤 padding
                if mask is not None:
                    valid_l = mask[:, :seq_l].reshape(-1).bool()
                    valid_a = mask[:, :seq_a].reshape(-1).bool()
                    aux_loss = criterion_aux(aux_pred_l[valid_l], labels_l[valid_l]) \
                             + criterion_aux(aux_pred_a[valid_a], labels_a[valid_a])
                else:
                    aux_loss = criterion_aux(aux_pred_l, labels_l) + criterion_aux(aux_pred_a, labels_a)
            else:
                criterion_aux = nn.L1Loss() if self.dataset in ['mosi', 'mosi-bert', 'mosei_senti', 'mosei-bert'] \
                    else nn.CrossEntropyLoss()
                aux_pred_l = self.aux_classifier_l(h_l.mean(dim=0))
                aux_pred_a = self.aux_classifier_a(h_a.mean(dim=0))
                aux_loss = criterion_aux(aux_pred_l, labels) + criterion_aux(aux_pred_a, labels)
                if h_v is not None:
                    aux_pred_v = self.aux_classifier_v(h_v.mean(dim=0))
                    aux_loss += criterion_aux(aux_pred_v, labels)

        # ==================== 5. 按 pattern 分组生成 ====================
        gen_loss = torch.tensor(0.0, device=device)
        ortho_loss_val = torch.tensor(0.0, device=device)
        gen_count = 0

        h_l_orig = h_l.clone()
        h_a_orig = h_a.clone()
        h_v_orig = h_v.clone() if h_v is not None else None

        h_orig_map = {'L': h_l_orig, 'A': h_a_orig, 'V': h_v_orig}
        raw_map = {'L': x_l_encoded, 'A': x_a, 'V': x_v}
        len_map = {'L': self.orig_l_len, 'A': self.orig_a_len, 'V': self.orig_v_len}

        x_l_work = x_l_encoded.clone()
        x_a_work = x_a.clone()
        x_v_work = x_v.clone() if x_v is not None else None
        work_map = {'L': x_l_work, 'A': x_a_work, 'V': x_v_work}

        for pat_id, (miss_mods, avail_mods) in self.PATTERN_DEFS.items():
            pat_mask = (m_type == pat_id)
            if not pat_mask.any():
                continue

            src_parts = []
            for mod in avail_mods:
                h_orig = h_orig_map.get(mod)
                if h_orig is not None:
                    src_parts.append(h_orig[:, pat_mask, :])
            if len(src_parts) == 0:
                continue
            src = torch.cat(src_parts, dim=0)

            for target_mod in miss_mods:
                if target_mod == 'V' and not self.has_video:
                    continue
                target_len = len_map[target_mod]
                target_raw = raw_map[target_mod]

                generated, z_c, z_s = self.generator(
                    src, target_mod, target_seq_len=target_len)
                real_target = target_raw[pat_mask].detach()
                gen_loss = gen_loss + F.mse_loss(generated, real_target)
                ortho_loss_val = ortho_loss_val + orthogonality_loss(z_c, z_s)
                gen_count += 1

                work_map[target_mod][pat_mask] = generated.float()

        if gen_count > 0:
            gen_loss = gen_loss / gen_count
            ortho_loss_val = ortho_loss_val / gen_count

        # 重编码
        if flags['L'].any():
            mask_l = flags['L'].bool()
            h_l_new = self._encode_modality(x_l_work, self.proj_l, self.unimodal_encoder_l,
                                             self.ln_l, self.l_len, batch_size, device)
            h_l = h_l.clone()
            h_l[:, mask_l, :] = h_l_new[:, mask_l, :]

        if flags['A'].any():
            mask_a = flags['A'].bool()
            h_a_new = self._encode_modality(x_a_work, self.proj_a, self.unimodal_encoder_a,
                                             self.ln_a, self.a_len, batch_size, device,
                                             noise_gate=self.noise_gate_a)
            h_a = h_a.clone()
            h_a[:, mask_a, :] = h_a_new[:, mask_a, :]

        if self.has_video and h_v is not None and flags['V'].any():
            mask_v = flags['V'].bool()
            h_v_new = self._encode_modality(x_v_work, self.proj_v, self.unimodal_encoder_v,
                                             self.ln_v, self.v_len, batch_size, device,
                                             noise_gate=self.noise_gate_v)
            h_v = h_v.clone()
            h_v[:, mask_v, :] = h_v_new[:, mask_v, :]

        # ==================== 6. Reliability Embedding ====================
        rel_l = self.reliability_emb(flags['L'].long())
        rel_a = self.reliability_emb(flags['A'].long())
        h_l = h_l + rel_l.unsqueeze(0)
        h_a = h_a + rel_a.unsqueeze(0)
        if h_v is not None:
            rel_v = self.reliability_emb(flags['V'].long())
            h_v = h_v + rel_v.unsqueeze(0)

        # ==================== 7. Missing-Type Embedding ====================
        mt_emb = self.missing_type_emb(m_type)
        h_l = h_l + mt_emb.unsqueeze(0) * self.mt_scale
        h_a = h_a + mt_emb.unsqueeze(0) * self.mt_scale
        if h_v is not None:
            h_v = h_v + mt_emb.unsqueeze(0) * self.mt_scale

        # ==================== 8. Uncertainty ====================
        if self.use_gate:
            # Adaptive Gate: 生成质量差 → 混入 default token
            if flags['L'].any():
                conf_l = self.uncertainty_head_l(h_l.mean(dim=0))
                mask_gen_l = flags['L'].unsqueeze(-1)
                gate_l = (mask_gen_l * (1.0 - conf_l)).unsqueeze(0)
                default_l = self.default_token_l.expand_as(h_l)
                h_l = h_l * (1.0 - gate_l) + default_l * gate_l

            if flags['A'].any():
                conf_a = self.uncertainty_head_a(h_a.mean(dim=0))
                mask_gen_a = flags['A'].unsqueeze(-1)
                gate_a = (mask_gen_a * (1.0 - conf_a)).unsqueeze(0)
                default_a = self.default_token_a.expand_as(h_a)
                h_a = h_a * (1.0 - gate_a) + default_a * gate_a

            if h_v is not None and flags['V'].any():
                conf_v = self.uncertainty_head_v(h_v.mean(dim=0))
                mask_gen_v = flags['V'].unsqueeze(-1)
                gate_v = (mask_gen_v * (1.0 - conf_v)).unsqueeze(0)
                default_v = self.default_token_v.expand_as(h_v)
                h_v = h_v * (1.0 - gate_v) + default_v * gate_v
        else:
            # 简单缩放: 只降低生成特征的权重
            if flags['L'].any():
                conf_l = self.uncertainty_head_l(h_l.mean(dim=0))
                mask_gen_l = flags['L'].unsqueeze(-1)
                h_l = h_l * (1.0 - mask_gen_l * (1.0 - conf_l)).unsqueeze(0)

            if flags['A'].any():
                conf_a = self.uncertainty_head_a(h_a.mean(dim=0))
                mask_gen_a = flags['A'].unsqueeze(-1)
                h_a = h_a * (1.0 - mask_gen_a * (1.0 - conf_a)).unsqueeze(0)

            if h_v is not None and flags['V'].any():
                conf_v = self.uncertainty_head_v(h_v.mean(dim=0))
                mask_gen_v = flags['V'].unsqueeze(-1)
                h_v = h_v * (1.0 - mask_gen_v * (1.0 - conf_v)).unsqueeze(0)

        # ==================== 9. CLS ====================
        scale = math.sqrt(self.embed_dim)
        cls_token = self.cls.unsqueeze(1).repeat(1, batch_size, 1) * scale

        # ==================== 10. Missing-Aware Dynamic Prompt [改进] ====================
        # 缺失上下文: missing_type_emb 编码当前缺失模式
        miss_ctx = self.missing_type_emb(m_type)  # (batch, embed_dim)

        # 每个模态: prompt = MLP(cat[summary, miss_ctx])
        summary_l = h_l.mean(dim=0)  # (batch, embed_dim)
        p_l = self._make_prompt(self.prompt_gen_l, summary_l, miss_ctx, batch_size)

        summary_a = h_a.mean(dim=0)
        p_a = self._make_prompt(self.prompt_gen_a, summary_a, miss_ctx, batch_size)

        seqs = [cls_token, p_l, h_l, p_a, h_a]
        ids = [
            torch.zeros(cls_token.shape[0], dtype=torch.long, device=device),
            torch.ones(p_l.shape[0], dtype=torch.long, device=device),
            torch.ones(h_l.shape[0], dtype=torch.long, device=device),
            torch.full((p_a.shape[0],), 2, dtype=torch.long, device=device),
            torch.full((h_a.shape[0],), 2, dtype=torch.long, device=device),
        ]

        if h_v is not None and hasattr(self, 'prompt_gen_v'):
            summary_v = h_v.mean(dim=0)
            p_v = self._make_prompt(self.prompt_gen_v, summary_v, miss_ctx, batch_size)
            seqs.extend([p_v, h_v])
            ids.extend([
                torch.full((p_v.shape[0],), 3, dtype=torch.long, device=device),
                torch.full((h_v.shape[0],), 3, dtype=torch.long, device=device),
            ])

        fused_input = torch.cat(seqs, dim=0)
        type_ids = torch.cat(ids, dim=0).unsqueeze(1).repeat(1, batch_size)
        fused_input = fused_input + self.fusion_modal_type_embeddings(type_ids)

        # ==================== 11. 融合 ====================
        final_fused = self.fusion_encoder(fused_input)

        # ==================== 12. 预测 ====================
        if self.is_meld:
            last_hs = final_fused[:cls_token.shape[0]]
        else:
            att_scores = self.att_pool_linear(final_fused)
            att_weights = F.softmax(att_scores, dim=0)
            last_hs = (att_weights * final_fused).sum(dim=0)

        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        if self.is_meld:
            output = output.transpose(0, 1)

        return output, last_hs, gen_loss, ortho_loss_val, aux_loss