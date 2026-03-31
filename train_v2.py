"""
Training Pipeline V3.1 + AMP加速
================================
加速方案（不影响性能）:
  1. AMP 混合精度训练 (float16) — 加速 40-60%
  2. Pretrain 方向 9→6（去掉双模态→单模态，主训练中会覆盖）
  3. zero_grad(set_to_none=True) — 减少内存操作

损失:
  L_total = L_cls + λ_gen * L_gen + λ_ortho * L_ortho + λ_aux * L_auxiliary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src import models_v2 as models
from src.generator import orthogonality_loss
from src.utils import save_model, load_model
from src.eval_metrics import eval_mosi, eval_mosei_senti, eval_iemocap, eval_sims


def unpack_batch(batch, hyp_params):
    """统一数据解包，兼容 MOSI/MOSEI 和 MELD 格式
    返回: text, audio, vision, labels, mask
    mask: (batch, seq) float tensor, 1=有效 0=padding; 非 MELD 时为 None
    """
    is_meld = hyp_params.dataset in ['meld_senti', 'meld_emo']
    if is_meld:
        # MELD TensorDataset: [text(33,600), audio(33,600), mask(33,), labels(33,)]
        text = batch[0]      # (batch, 33, 600)
        audio = batch[1]     # (batch, 33, 600)
        mask = batch[2]      # (batch, 33) float 0/1
        labels = batch[3].long()  # (batch, 33) int64
        vision = None
        if hyp_params.use_cuda:
            text, audio, labels, mask = text.cuda(), audio.cuda(), labels.cuda(), mask.cuda()
        return text, audio, vision, labels, mask
    else:
        # MOSI/MOSEI: ((sample_ind, text, audio, vision), labels, meta)
        batch_X, batch_Y = batch[0], batch[1]
        _, text, audio, vision = batch_X
        labels = batch_Y.squeeze(-1)
        if hyp_params.use_cuda:
            text, audio, vision, labels = text.cuda(), audio.cuda(), vision.cuda(), labels.cuda()
            if hyp_params.dataset == 'iemocap':
                labels = labels.long()
        return text, audio, vision, labels, None


def eval_meld(results, truths):
    """MELD 评估: Accuracy + Weighted F1"""
    from sklearn.metrics import f1_score, accuracy_score
    if results.dim() == 3:
        results = results.reshape(-1, results.size(-1))
        truths = truths.reshape(-1)
    preds = results.argmax(dim=-1).cpu().numpy()
    labels = truths.cpu().numpy()
    mask = labels >= 0
    preds = preds[mask]
    labels = labels[mask]
    wf1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    print(f"  Weighted F1: {wf1*100:.2f}%")
    print(f"  Accuracy:    {acc*100:.2f}%")
    print("-" * 50)
    return acc  # 返回 accuracy，和 UniMF 保持一致


def pretrain_generator(model, train_loader, hyp_params, epochs=10):
    print("=" * 60)
    print("Stage 1: Pretraining Generator")
    directions = 6 if model.has_video else 2
    print(f"  Epochs: {epochs} | Directions: {directions}")
    print("=" * 60)

    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(k in name for k in ['generator', 'proj_l', 'proj_a', 'proj_v',
                                     'unimodal_encoder', 'ln_l', 'ln_a', 'ln_v',
                                     'noise_gate', 'pos_enc']):
            param.requires_grad = True

    gen_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(gen_params, lr=1e-3)
    scaler = GradScaler(enabled=hyp_params.use_cuda)
    print(f"  Trainable params: {sum(p.numel() for p in gen_params):,}")

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, total_gen, total_ortho, n = 0, 0, 0, 0

        for batch in train_loader:
            text, audio, vision, _, mask = unpack_batch(batch, hyp_params)

            bs = text.size(0)
            device = text.device
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=hyp_params.use_cuda):
                if hyp_params.use_bert:
                    with torch.no_grad():
                        text_enc = model.text_model(text)
                else:
                    text_enc = text

                h_l = model._encode_modality(text_enc, model.proj_l, model.unimodal_encoder_l,
                                              model.ln_l, model.l_len, bs, device)
                h_a = model._encode_modality(audio, model.proj_a, model.unimodal_encoder_a,
                                              model.ln_a, model.a_len, bs, device,
                                              noise_gate=model.noise_gate_a)

                gen_loss = torch.tensor(0., device=device)
                ortho = torch.tensor(0., device=device)
                count = 0
                text_target = text_enc

                def add_direction(src, target_mod, target_raw, target_len):
                    nonlocal gen_loss, ortho, count
                    g, zc, zs = model.generator(src, target_mod, target_len)
                    gen_loss = gen_loss + F.mse_loss(g, target_raw)
                    ortho = ortho + orthogonality_loss(zc, zs)
                    count += 1

                # 单模态→单模态 (6方向，去掉双模态→单模态)
                add_direction(h_l, 'A', audio, model.orig_a_len)
                add_direction(h_a, 'L', text_target, model.orig_l_len)

                if model.has_video:
                    h_v = model._encode_modality(vision, model.proj_v, model.unimodal_encoder_v,
                                                  model.ln_v, model.v_len, bs, device,
                                                  noise_gate=model.noise_gate_v)
                    add_direction(h_l, 'V', vision, model.orig_v_len)
                    add_direction(h_v, 'L', text_target, model.orig_l_len)
                    add_direction(h_a, 'V', vision, model.orig_v_len)
                    add_direction(h_v, 'A', audio, model.orig_a_len)

                gen_loss = gen_loss / count
                ortho = ortho / count
                loss = gen_loss + 0.1 * ortho

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * bs
            total_gen += gen_loss.item() * bs
            total_ortho += ortho.item() * bs
            n += bs

        print(f"  Epoch {epoch}/{epochs} | Loss: {total_loss/n:.4f} "
              f"| Gen: {total_gen/n:.4f} | Ortho: {total_ortho/n:.4f}")

    for param in model.parameters():
        param.requires_grad = True
    print("  Generator pretraining complete.\n")
    return model


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = models.CFMModel(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()

    # 加载预训练权重（如果有）
    pretrained_path = getattr(hyp_params, 'pretrained_model', '')
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"  Loading pretrained model from: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path,
                                      map_location='cuda' if hyp_params.use_cuda else 'cpu')
        model_dict = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(model_dict)
        print(f"  Loaded {loaded} params, skipped {skipped} (shape mismatch)")

    pretrain_epochs = getattr(hyp_params, 'pretrain_epochs', 10)
    if pretrain_epochs > 0:
        model = pretrain_generator(model, train_loader, hyp_params, epochs=pretrain_epochs)

    # 优化器
    if hyp_params.use_bert:
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.text_model.named_parameters())
        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        gen_params = [p for n, p in model.named_parameters()
                      if 'generator' in n and 'text_model' not in n]
        other_params = [p for n, p in model.named_parameters()
                        if 'text_model' not in n and 'generator' not in n]
        optimizer = optim.Adam([
            {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert,
             'lr': hyp_params.lr_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
            {'params': gen_params, 'weight_decay': 0.0, 'lr': hyp_params.lr * 0.1},
            {'params': other_params, 'weight_decay': 0.0, 'lr': hyp_params.lr}
        ])
    else:
        gen_params = [p for n, p in model.named_parameters() if 'generator' in n]
        other_params = [p for n, p in model.named_parameters() if 'generator' not in n]
        optimizer = optim.Adam([
            {'params': other_params, 'lr': hyp_params.lr, 'weight_decay': 1e-4},
            {'params': gen_params, 'lr': hyp_params.lr * 0.1, 'weight_decay': 0}
        ])

    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)

    return train_model(
        {'model': model, 'optimizer': optimizer, 'criterion': criterion, 'scheduler': scheduler},
        hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    lambda_gen = getattr(hyp_params, 'lambda_gen', 0.5)
    lambda_ortho = getattr(hyp_params, 'lambda_ortho', 0.1)
    lambda_aux = getattr(hyp_params, 'lambda_aux', 0.3)

    # AMP
    scaler = GradScaler(enabled=hyp_params.use_cuda)

    def train_epoch():
        model.train()
        e_cls, e_gen, e_ortho, e_aux, n_samples = 0, 0, 0, 0, 0

        for batch in train_loader:
            text, audio, vision, eval_attr, mask = unpack_batch(batch, hyp_params)

            bs = text.size(0)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=hyp_params.use_cuda):
                net = nn.DataParallel(model) if hyp_params.distribute else model
                preds, _, gen_loss, ortho_loss, aux_loss = net(
                    text, audio, vision, phase='train', labels=eval_attr, mask=mask)

                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                elif hyp_params.dataset in ['meld_senti', 'meld_emo']:
                    preds = preds.reshape(-1, preds.size(-1))
                    eval_attr = eval_attr.reshape(-1)
                    # 用 mask 只保留有效 utterance
                    valid = mask.reshape(-1).bool()
                    preds = preds[valid]
                    eval_attr = eval_attr[valid]

                cls_loss = criterion(preds, eval_attr)

                gen_loss = gen_loss.mean() if gen_loss.dim() > 0 else gen_loss
                ortho_loss = ortho_loss.mean() if ortho_loss.dim() > 0 else ortho_loss
                aux_loss = aux_loss.mean() if aux_loss.dim() > 0 else aux_loss

                total = (cls_loss
                         + lambda_gen * gen_loss
                         + lambda_ortho * ortho_loss
                         + lambda_aux * aux_loss)

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            scaler.step(optimizer)
            scaler.update()

            e_cls += cls_loss.item() * bs
            e_gen += gen_loss.item() * bs
            e_ortho += ortho_loss.item() * bs
            e_aux += aux_loss.item() * bs
            n_samples += bs

        return (e_cls/n_samples, e_gen/n_samples, e_ortho/n_samples, e_aux/n_samples)

    def evaluate(test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss, results, truths = 0, [], []
        with torch.no_grad():
            for batch in loader:
                text, audio, vision, eval_attr, mask = unpack_batch(batch, hyp_params)

                with autocast(enabled=hyp_params.use_cuda):
                    net = nn.DataParallel(model) if hyp_params.distribute else model
                    preds, _, _, _, _ = net(text, audio, vision, phase='valid')

                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                elif hyp_params.dataset in ['meld_senti', 'meld_emo']:
                    preds = preds.reshape(-1, preds.size(-1))
                    eval_attr = eval_attr.reshape(-1)
                    valid = mask.reshape(-1).bool()
                    preds = preds[valid]
                    eval_attr = eval_attr[valid]

                total_loss += criterion(preds.float(), eval_attr).item() * preds.size(0)
                results.append(preds.float())
                truths.append(eval_attr)

        avg_loss = total_loss / max(1, sum(r.size(0) for r in results))
        return avg_loss, torch.cat(results), torch.cat(truths)

    def evaluate_missing(missing_modalities):
        model.eval()
        results, truths = [], []
        with torch.no_grad():
            for batch in test_loader:
                text, audio, vision, eval_attr, mask = unpack_batch(batch, hyp_params)

                bs = text.size(0)
                md = model.module.modality_dropout if hasattr(model, 'module') else model.modality_dropout
                miss_flags, miss_t = md.set_test_missing(bs, text.device, missing_modalities)

                with autocast(enabled=hyp_params.use_cuda):
                    net = nn.DataParallel(model) if hyp_params.distribute else model
                    preds, _, _, _, _ = net(text, audio, vision, phase='test',
                                            missing_flags=miss_flags, miss_type=miss_t)
                if hyp_params.dataset in ['meld_senti', 'meld_emo']:
                    preds = preds.reshape(-1, preds.size(-1))
                    eval_attr = eval_attr.reshape(-1)
                    valid = mask.reshape(-1).bool()
                    preds = preds[valid]
                    eval_attr = eval_attr[valid]
                results.append(preds.float())
                truths.append(eval_attr)
        return torch.cat(results), torch.cat(truths)

    def eval_dataset(results, truths):
        if hyp_params.dataset in ["mosei_senti", 'mosei-bert']:
            return eval_mosei_senti(results, truths, True)
        elif hyp_params.dataset in ['mosi', 'mosi-bert']:
            return eval_mosi(results, truths, True)
        elif hyp_params.dataset in ['meld_senti', 'meld_emo']:
            return eval_meld(results, truths)
        elif hyp_params.dataset == 'iemocap':
            return eval_iemocap(results, truths)
        elif hyp_params.dataset == 'sims':
            return eval_sims(results, truths)
        return 0.0

    # ==================== 主训练循环 ====================
    print("=" * 60)
    print("Stage 2: Main Training (AMP enabled)")
    print("=" * 60)
    total_p = sum(p.numel() for p in model.parameters())
    gen_p = sum(p.numel() for n, p in model.named_parameters() if 'generator' in n)
    print(f"  Total: {total_p:,} | Generator: {gen_p:,}")
    print(f"  Weights: cls=1.0, gen={lambda_gen:.3f}, ortho={lambda_ortho:.3f}, aux={lambda_aux:.3f}")

    best_valid = 1e8
    loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)
    for epoch in loop:
        loop.set_description(f'Epoch {epoch}/{hyp_params.num_epochs}')
        cls, gen, ort, aux = train_epoch()
        val_loss, _, _ = evaluate(test=False)
        scheduler.step(val_loss)
        loop.set_postfix(cls=f'{cls:.3f}', gen=f'{gen:.3f}', aux=f'{aux:.3f}',
                         val=f'{val_loss:.3f}')
        if val_loss < best_valid:
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    # ==================== 测试 ====================
    model = load_model(hyp_params, name=hyp_params.name)
    full_mod = hyp_params.modalities  # 'LAV' or 'LA'

    print("\n" + "=" * 60)
    print(f"Full Modality ({full_mod})")
    print("=" * 60)
    _, results, truths = evaluate(test=True)
    acc = eval_dataset(results, truths)

    # 收集所有场景结果
    all_results = {f'Full({full_mod})': {'acc': acc}}

    if model.has_video:
        scenarios = [('A', 'LV'), ('V', 'LA'), ('AV', 'L'),
                     ('L', 'AV'), ('LV', 'A'), ('LA', 'V')]
    else:
        scenarios = [('A', 'L'), ('L', 'A')]

    for missing, available in scenarios:
        print(f"\nMissing {missing} (available: {available})")
        print("-" * 40)
        mr, mt = evaluate_missing(missing)
        sc_acc = eval_dataset(mr, mt)
        all_results[f'-{missing}({available})'] = {'acc': sc_acc}

    # 保存日志
    _save_trial_log(hyp_params, all_results)

    return acc


def _save_trial_log(hyp_params, all_results):
    """所有 trial 结果追加到同一个 CSV 文件"""
    from datetime import datetime

    os.makedirs('logs', exist_ok=True)
    dataset = hyp_params.dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    is_meld = dataset in ['meld_senti', 'meld_emo']

    summary_file = f'logs/{dataset}_all_trials.csv'
    write_header = not os.path.exists(summary_file)

    with open(summary_file, 'a') as f:
        if write_header:
            if is_meld:
                headers = [
                    'time', 'Full(LA)', '-A(L)', '-L(A)', 'Avg3',
                    'miss_prob', 'lambda_gen', 'lambda_aux',
                    'ml', 'ak', 'attn_drop', 'embed_drop', 'out_drop'
                ]
            else:
                headers = [
                    'time', 'Full(LAV)', '-A(LV)', '-V(LA)', '-AV(L)',
                    '-L(AV)', '-LV(A)', '-LA(V)', 'Avg7',
                    'miss_prob', 'lambda_gen', 'lambda_aux',
                    'ml', 'ak', 'vk', 'attn_drop', 'embed_drop', 'out_drop'
                ]
            f.write(','.join(headers) + '\n')

        accs = []
        for scenario, metrics in all_results.items():
            a = metrics.get('acc', 0)
            val = a * 100 if isinstance(a, float) and a < 1.5 else (a if a else 0)
            accs.append(val)

        avg = sum(accs) / len(accs) if accs else 0

        vals = [timestamp]
        vals += [f"{a:.2f}" for a in accs]
        vals.append(f"{avg:.2f}")
        vals += [
            f"{getattr(hyp_params, 'miss_prob', 0):.4f}",
            f"{getattr(hyp_params, 'lambda_gen', 0):.4f}",
            f"{getattr(hyp_params, 'lambda_aux', 0):.4f}",
            str(getattr(hyp_params, 'multimodal_layers', '')),
            str(getattr(hyp_params, 'a_kernel_size', '')),
        ]
        if not is_meld:
            vals.append(str(getattr(hyp_params, 'v_kernel_size', '')))
        vals += [
            f"{getattr(hyp_params, 'attn_dropout', 0):.4f}",
            f"{getattr(hyp_params, 'embed_dropout', 0):.4f}",
            f"{getattr(hyp_params, 'out_dropout', 0):.4f}",
        ]
        f.write(','.join(vals) + '\n')

    print(f"\n  Results appended to: {summary_file}")