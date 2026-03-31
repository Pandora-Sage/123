"""
run_fixed.py - 固定最优超参，跑多次取 mean±std
用法:
  python run_fixed.py --dataset=mosi --modalities=LAV --runs=5
  python run_fixed.py --dataset=mosei_senti --modalities=LAV --runs=5
"""

import torch
import argparse
import os
import numpy as np
from src.utils import *
from torch.utils.data import DataLoader
from src import train_v2
from bert_dataloader import MMDataset

parser = argparse.ArgumentParser(description='Fixed Hyperparameter Evaluation')
parser.add_argument('-f', default='', type=str)
parser.add_argument('--model', type=str, default='CFM')
parser.add_argument('--aligned', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='mosi',
                    choices=['mosi', 'mosei_senti', 'mosi-bert', 'mosei-bert',
                             'meld_senti', 'meld_emo'])
parser.add_argument('--data_path', type=str, default='data')
parser.add_argument('--runs', type=int, default=5, help='Number of runs with different seeds')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--clip', type=float, default=0.8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--when', type=int, default=20)
parser.add_argument('--no_cuda', type=bool, default=False)
parser.add_argument('--distribute', action='store_true')
parser.add_argument('--name', type=str, default='UniMF_V3')
parser.add_argument('--use_bert', action='store_true')
parser.add_argument('--language', type=str, default='en')
parser.add_argument('--modalities', type=str, default='LAV')
parser.add_argument('--embed_dim', type=int, default=32)
parser.add_argument('--prompt_len', type=int, default=4)
parser.add_argument('--pretrain_epochs', type=int, default=10)
parser.add_argument('--pretrained_model', type=str, default='',
                    help='MOSEI 预训练模型路径')
parser.add_argument('--use_gate', type=int, default=1,
                    help='1=Adaptive Gate, 0=简单缩放')

# ============================================================
# 最优超参（按数据集自动选择）
# ============================================================
BEST_PARAMS = {
    'mosi': {
        'attn_dropout': 0.2897, 'embed_dropout': 0.4321, 'out_dropout': 0.3712,
        'l_kernel_size': 1, 'a_kernel_size': 2, 'v_kernel_size': 1,
        'multimodal_layers': 4,
        'miss_prob': 0.6095, 'lambda_gen': 0.3625, 'lambda_aux': 0.2598,
    },
    'mosei_senti': {
        'attn_dropout': 0.4353, 'embed_dropout': 0.3253, 'out_dropout': 0.3320,
        'l_kernel_size': 1, 'a_kernel_size': 2, 'v_kernel_size': 1,
        'multimodal_layers': 2,
        'miss_prob': 0.3975, 'lambda_gen': 0.4535, 'lambda_aux': 0.4273,
    },
    'meld_senti': {
        'attn_dropout': 0.4594, 'embed_dropout': 0.4293, 'out_dropout': 0.1599,
        'l_kernel_size': 1, 'a_kernel_size': 4,
        'multimodal_layers': 2,
        'miss_prob': 0.5479, 'lambda_gen': 0.6002, 'lambda_aux': 0.3297,
    },
}

# 通用默认参数
parser.add_argument('--attn_dropout', type=float, default=0.3)
parser.add_argument('--embed_dropout', type=float, default=0.4)
parser.add_argument('--out_dropout', type=float, default=0.35)
parser.add_argument('--relu_dropout', type=float, default=0.1)
parser.add_argument('--res_dropout', type=float, default=0.1)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--l_kernel_size', type=int, default=1)
parser.add_argument('--a_kernel_size', type=int, default=2)
parser.add_argument('--v_kernel_size', type=int, default=1)
parser.add_argument('--multimodal_layers', type=int, default=3)
parser.add_argument('--miss_prob', type=float, default=0.4)
parser.add_argument('--lambda_gen', type=float, default=0.5)
parser.add_argument('--lambda_ortho', type=float, default=0.1)
parser.add_argument('--lambda_aux', type=float, default=0.3)

args = parser.parse_args()

# 自动覆盖为最优超参
dataset_key = str.lower(args.dataset.strip())
if dataset_key in BEST_PARAMS:
    bp = BEST_PARAMS[dataset_key]
    for k, v in bp.items():
        setattr(args, k, v)
    print(f"  Loaded best params for {dataset_key}")
else:
    print(f"  Warning: no best params for {dataset_key}, using defaults")

output_dim_dict = {
    'mosi': 1, 'mosei_senti': 1, 'iemocap': 8,
    'meld_senti': 3, 'meld_emo': 7,
    'mosi-bert': 1, 'mosei-bert': 1, 'sims': 1
}
criterion_dict = {
    'mosi': 'L1Loss', 'mosei_senti': 'L1Loss', 'iemocap': 'CrossEntropyLoss',
    'meld_senti': 'CrossEntropyLoss', 'meld_emo': 'CrossEntropyLoss',
    'mosi-bert': 'L1Loss', 'mosei-bert': 'L1Loss', 'sims': 'L1Loss'
}

dataset = str.lower(args.dataset.strip())

# ============================================================
# 跑多次
# ============================================================
seeds = [1111, 2222, 3333, 4444, 5555][:args.runs]
all_accs = []

for run_idx, seed in enumerate(seeds):
    print(f"\n{'='*60}")
    print(f"  Run {run_idx+1}/{args.runs} | Seed: {seed}")
    print(f"{'='*60}")

    args.seed = seed
    seed_everything(args)

    use_cuda = False
    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available() and not args.no_cuda:
        seed_everything(args)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

    # 加载数据
    if dataset in ['mosi', 'mosei_senti', 'iemocap']:
        train_data = get_data(args, dataset, 'train')
        valid_data = get_data(args, dataset, 'valid')
        test_data = get_data(args, dataset, 'test')
    elif dataset in ['mosi-bert', 'mosei-bert', 'sims']:
        args.use_bert = True
        args.language = 'cn' if dataset == 'sims' else 'en'
        args.weight_decay_bert = 0.001
        args.lr_bert = 5e-5
        train_data = MMDataset(args, 'train')
        valid_data = MMDataset(args, 'valid')
        test_data = MMDataset(args, 'test')
    elif dataset in ['meld_senti', 'meld_emo']:
        import os
        from src.utils import load_meld
        mode_name = 'sentiment' if dataset == 'meld_senti' else 'emotion'
        classes, meld_info, train_data, valid_data, test_data, \
            mask_train, mask_valid, mask_test = load_meld(mode_name, os.path.join(args.data_path, 'MELD/'))
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    gen_device = 'cuda' if use_cuda else 'cpu'
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator(device=gen_device))
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator(device=gen_device))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True,
                             generator=torch.Generator(device=gen_device))

    # 超参设置
    hyp_params = args
    hyp_params.use_cuda = use_cuda
    hyp_params.dataset = dataset

    if dataset in ['mosi', 'mosei_senti', 'iemocap']:
        hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
        hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
    elif dataset == 'meld_senti':
        hyp_params.orig_d_l, hyp_params.orig_d_a = 600, 600
        hyp_params.l_len, hyp_params.a_len = 33, 33
        hyp_params.orig_d_v, hyp_params.v_len = 0, 0
    elif dataset == 'meld_emo':
        hyp_params.orig_d_l, hyp_params.orig_d_a = 300, 600
        hyp_params.l_len, hyp_params.a_len = 33, 33
        hyp_params.orig_d_v, hyp_params.v_len = 0, 0
    elif dataset == 'mosi-bert':
        hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 5, 20
        if args.aligned:
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 50, 50
        else:
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 375, 500
    elif dataset == 'mosei-bert':
        hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 74, 35
        if args.aligned:
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 50, 50
        else:
            hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 500, 500

    hyp_params.n_train = len(train_data)
    hyp_params.n_valid = len(valid_data)
    hyp_params.n_test = len(test_data)
    hyp_params.model = 'CFM'
    hyp_params.output_dim = output_dim_dict.get(dataset, 1)
    hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')

    if 'meld' in dataset and hyp_params.modalities == 'LAV':
        hyp_params.modalities = 'LA'

    print(f"  miss_prob={hyp_params.miss_prob:.4f} | lambda_gen={hyp_params.lambda_gen:.4f} "
          f"| lambda_aux={hyp_params.lambda_aux:.4f}")

    # 训练
    acc = train_v2.initiate(hyp_params, train_loader, valid_loader, test_loader)
    all_accs.append(acc)

    print(f"\n  >>> Run {run_idx+1} Accuracy: {acc*100:.2f}%")

# ============================================================
# 汇总结果
# ============================================================
print("\n" + "=" * 60)
print("  FINAL RESULTS")
print("=" * 60)
accs_pct = [a * 100 for a in all_accs]
for i, (s, a) in enumerate(zip(seeds, accs_pct)):
    print(f"  Run {i+1} (seed={s}): {a:.2f}%")
print(f"\n  Mean: {np.mean(accs_pct):.2f}%")
print(f"  Std:  {np.std(accs_pct):.2f}%")
print(f"  Result: {np.mean(accs_pct):.2f} ± {np.std(accs_pct):.2f}")
print("=" * 60)

# 保存结果
os.makedirs('results', exist_ok=True)
with open(f'results/{dataset}_fixed_results.txt', 'w') as f:
    f.write(f"Dataset: {dataset}\n")
    f.write(f"Runs: {args.runs}\n")
    f.write(f"Hyperparameters:\n")
    f.write(f"  miss_prob={args.miss_prob}, lambda_gen={args.lambda_gen}, lambda_aux={args.lambda_aux}\n")
    f.write(f"  attn_dropout={args.attn_dropout}, embed_dropout={args.embed_dropout}, out_dropout={args.out_dropout}\n")
    f.write(f"  multimodal_layers={args.multimodal_layers}, a_kernel_size={args.a_kernel_size}, v_kernel_size={args.v_kernel_size}\n\n")
    for i, (s, a) in enumerate(zip(seeds, accs_pct)):
        f.write(f"Run {i+1} (seed={s}): {a:.2f}%\n")
    f.write(f"\nMean: {np.mean(accs_pct):.2f}%\n")
    f.write(f"Std:  {np.std(accs_pct):.2f}%\n")
    f.write(f"Result: {np.mean(accs_pct):.2f} ± {np.std(accs_pct):.2f}\n")
print(f"\nResults saved to results/{dataset}_fixed_results.txt")