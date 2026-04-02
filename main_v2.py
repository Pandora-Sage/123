"""
main_v2.py - UniMF V3.1 入口
============================
去掉 lambda_con 参数
"""

import torch
import argparse
import os
from src.utils import *
from torch.utils.data import DataLoader
from src import train_v2
from bert_dataloader import MMDataset

import optuna
from optuna.samplers import TPESampler

parser = argparse.ArgumentParser(description='UniMF V3.1')
parser.add_argument('-f', default='', type=str)
parser.add_argument('--model', type=str, default='CFM')
parser.add_argument('--aligned', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='mosi',
                    choices=['mosi', 'mosei_senti', 'mosi-bert', 'mosei-bert',
                             'meld_senti', 'meld_emo', 'urfunny'])
parser.add_argument('--data_path', type=str, default='data')
parser.add_argument('--run_id', type=int, default=1)
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--relu_dropout', type=float, default=0.1)
parser.add_argument('--res_dropout', type=float, default=0.1)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--clip', type=float, default=0.8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--when', type=int, default=20)
parser.add_argument('--log_interval', type=int, default=30)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--no_cuda', type=bool, default=False)
parser.add_argument('--distribute', action='store_true')
parser.add_argument('--name', type=str, default='UniMF_V3')
parser.add_argument('--use_bert', action='store_true')
parser.add_argument('--language', type=str, default='en')
parser.add_argument('--modalities', type=str, default='LAV',
                    choices=['LAV', 'LA', 'LV', 'AV', 'L', 'A', 'V'])

# V3.1 参数（去掉 lambda_con）
parser.add_argument('--embed_dim', type=int, default=32)
parser.add_argument('--miss_prob', type=float, default=0.4)
parser.add_argument('--lambda_gen', type=float, default=0.5)
parser.add_argument('--lambda_ortho', type=float, default=0.1)
parser.add_argument('--lambda_aux', type=float, default=0.3,
                    help='辅助分类损失权重')
parser.add_argument('--prompt_len', type=int, default=4,
                    help='Modality-Aware Prompt 长度')
parser.add_argument('--pretrain_epochs', type=int, default=10)
parser.add_argument('--pretrained_model', type=str, default='',
                    help='MOSEI 预训练模型路径，如 pretrained/mosei.pt')
parser.add_argument('--use_gate', type=int, default=1,
                    help='1=Adaptive Gate, 0=简单缩放')

args = parser.parse_args()
seed_everything(args)
dataset = str.lower(args.dataset.strip())
use_cuda = False

output_dim_dict = {
    'mosi': 1, 'mosei_senti': 1, 'iemocap': 8,
    'meld_senti': 3, 'meld_emo': 7, 'urfunny': 1,
    'mosi-bert': 1, 'mosei-bert': 1, 'sims': 1
}
criterion_dict = {
    'mosi': 'L1Loss', 'mosei_senti': 'L1Loss', 'iemocap': 'CrossEntropyLoss',
    'meld_senti': 'CrossEntropyLoss', 'meld_emo': 'CrossEntropyLoss',
    'urfunny': 'BCEWithLogitsLoss', 'mosi-bert': 'L1Loss',
    'mosei-bert': 'L1Loss', 'sims': 'L1Loss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available() and not args.no_cuda:
    seed_everything(args)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    use_cuda = True

# ======================== 加载数据 ========================
print("Start loading the data....")
if dataset in ['mosi', 'mosei_senti', 'iemocap']:
    if args.use_bert:
        raise ValueError('For BERT, use mosi-bert or mosei-bert')
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
    mode_name = 'sentiment' if dataset == 'meld_senti' else 'emotion'
    classes, meld_info, train_data, valid_data, test_data, \
        mask_train, mask_valid, mask_test = load_meld(mode_name, os.path.join(args.data_path, 'MELD/'))
else:
    raise ValueError(f'Unknown dataset: {dataset}')

gen_device = 'cuda' if use_cuda else 'cpu'

# MELD 数据量小(1038)，用小 batch 增加梯度更新次数，延长训练
if dataset in ['meld_senti', 'meld_emo']:
    args.batch_size = 32    # 128→32: 每epoch 32步(原来8步)
    args.num_epochs = 150   # 100→150: 更充分训练

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                          generator=torch.Generator(device=gen_device))
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True,
                          generator=torch.Generator(device=gen_device))
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True,
                         generator=torch.Generator(device=gen_device))

print('Finish loading the data....')
print(f'### Dataset: {str.upper(dataset)}')

# ======================== 超参 ========================
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
    hyp_params.orig_d_l, hyp_params.orig_d_a = 600, 300
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
elif dataset == 'sims':
    hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768, 33, 709
    hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 39, 400, 55

hyp_params.n_train = len(train_data)
hyp_params.n_valid = len(valid_data)
hyp_params.n_test = len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')

if 'meld' in dataset and hyp_params.modalities == 'LAV':
    hyp_params.modalities = 'LA'

print(f'### Modalities: {hyp_params.modalities} | embed_dim: {hyp_params.embed_dim}')
print(f'### miss_prob: {hyp_params.miss_prob} | prompt_len: {hyp_params.prompt_len}')
print(f'### Losses: gen={hyp_params.lambda_gen}, ortho={hyp_params.lambda_ortho}, aux={hyp_params.lambda_aux}')
if hyp_params.use_bert:
    print(f'### BERT: ON')

# ======================== Optuna ========================
if __name__ == '__main__':
    sampler = TPESampler(seed=args.seed)

    def objective(trial):
        if dataset in ['meld_senti', 'meld_emo']:
            # MELD: 缩窄到历史最优区间
            hyp_params.embed_dim = 64
            hyp_params.a_kernel_size = 2
            hyp_params.l_kernel_size = 1
            hyp_params.multimodal_layers = trial.suggest_int('multimodal_layers', 2, 3)
            hyp_params.prompt_len = trial.suggest_categorical('prompt_len', [2, 4])
            hyp_params.miss_prob = trial.suggest_float('miss_prob', 0.28, 0.45)
            hyp_params.lambda_gen = trial.suggest_float('lambda_gen', 0.40, 0.70)
            hyp_params.lambda_aux = trial.suggest_float('lambda_aux', 0.15, 0.45)
            hyp_params.attn_dropout = trial.suggest_float('attn_dropout', 0.10, 0.30)
            hyp_params.embed_dropout = trial.suggest_float('embed_dropout', 0.30, 0.45)
            hyp_params.out_dropout = trial.suggest_float('out_dropout', 0.15, 0.40)
            hyp_params.pretrain_epochs = 20
        else:
            # MOSI/MOSEI: 保持原搜索范围
            hyp_params.embed_dim = trial.suggest_categorical('embed_dim', [32, 64])
            hyp_params.prompt_len = trial.suggest_categorical('prompt_len', [2, 4, 8])
            hyp_params.l_kernel_size = 1
            hyp_params.a_kernel_size = trial.suggest_int('a_kernel_size', 2, 4, step=2)
            hyp_params.v_kernel_size = trial.suggest_int('v_kernel_size', 1, 3, step=2)
            hyp_params.multimodal_layers = trial.suggest_int('multimodal_layers', 2, 6)
            hyp_params.miss_prob = trial.suggest_float('miss_prob', 0.3, 0.7)
            hyp_params.lambda_gen = trial.suggest_float('lambda_gen', 0.2, 0.8)
            hyp_params.lambda_aux = trial.suggest_float('lambda_aux', 0.05, 0.5)
            hyp_params.attn_dropout = trial.suggest_float('attn_dropout', 0.1, 0.5)
            hyp_params.embed_dropout = trial.suggest_float('embed_dropout', 0.2, 0.5)
            hyp_params.out_dropout = trial.suggest_float('out_dropout', 0.1, 0.45)

        acc = train_v2.initiate(hyp_params, train_loader, valid_loader, test_loader)
        return acc

    n_trials = hyp_params.trials
    study = optuna.create_study(direction='maximize', sampler=sampler,
                                study_name='exp_v31_' + str(hyp_params.run_id))
    study.optimize(objective, n_trials=n_trials)

    print('-' * 60)
    print(f'Best Accuracy: {study.best_trial.value:.4f}')
    print(f'Best Params:\n{study.best_trial.params}')

    os.makedirs('results', exist_ok=True)
    alignment = 'aligned' if hyp_params.aligned else 'unaligned'
    study.trials_dataframe().to_csv(
        f'results/{dataset}_{alignment}_{hyp_params.modalities}_v31_trials_{n_trials}.csv')