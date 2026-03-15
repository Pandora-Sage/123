import torch
import torch.nn.functional as F
from torch import nn
import sys
import csv
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    if hyp_params.modalities != 'LAV':
        if hyp_params.modalities == 'L':
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            translator1_optimizer = getattr(optim, hyp_params.optim)(translator1.parameters(), lr=hyp_params.lr)
            translator2_optimizer = getattr(optim, hyp_params.optim)(translator2.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'A':
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            translator1_optimizer = getattr(optim, hyp_params.optim)(translator1.parameters(), lr=hyp_params.lr)
            translator2_optimizer = getattr(optim, hyp_params.optim)(translator2.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'V':
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            translator1_optimizer = getattr(optim, hyp_params.optim)(translator1.parameters(), lr=hyp_params.lr)
            translator2_optimizer = getattr(optim, hyp_params.optim)(translator2.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'LA':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'LV':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        elif hyp_params.modalities == 'AV':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        else:
            raise ValueError('Unknown modalities type')
        trans_criterion = getattr(nn, 'MSELoss')()
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    if hyp_params.use_bert:
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.text_model.named_parameters())
        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]
        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert, 'lr': hyp_params.lr_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
            {'params': model_params_other, 'weight_decay': 0.0, 'lr': hyp_params.lr}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
    else:
        optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)
    if hyp_params.modalities != 'LAV':
        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
            settings = {'model': model,
                        'translator1': translator1,
                        'translator2': translator2,
                        'translator1_optimizer': translator1_optimizer,
                        'translator2_optimizer': translator2_optimizer,
                        'trans_criterion': trans_criterion,
                        'optimizer': optimizer,
                        'criterion': criterion,
                        'scheduler': scheduler}
        elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
            settings = {'model': model,
                        'translator': translator,
                        'translator_optimizer': translator_optimizer,
                        'trans_criterion': trans_criterion,
                        'optimizer': optimizer,
                        'criterion': criterion,
                        'scheduler': scheduler}
        else:
            raise ValueError('Unknown modalities type')
    elif hyp_params.modalities == 'LAV':
        settings = {'model': model,
                    'optimizer': optimizer,
                    'criterion': criterion,
                    'scheduler': scheduler}
    else:
        raise ValueError('Unknown modalities type')
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    global acc
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    scheduler = settings['scheduler']

    if hyp_params.modalities != 'LAV':
        trans_criterion = settings['trans_criterion']
        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
            translator1 = settings['translator1']
            translator2 = settings['translator2']
            translator1_optimizer = settings['translator1_optimizer']
            translator2_optimizer = settings['translator2_optimizer']
            translator = (translator1, translator2)
        elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
            translator = settings['translator']
            translator_optimizer = settings['translator_optimizer']
        else:
            raise ValueError('Unknown modalities type')
    else:
        translator = None

    def train(model, translator, optimizer, criterion):
        if isinstance(translator, tuple):
            translator1, translator2 = translator
        epoch_loss = 0
        model.train()
        if hyp_params.modalities != 'LAV':
            if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                translator1.train()
                translator2.train()
            elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                translator.train()
            else:
                raise ValueError('Unknown modalities type')
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)  # if num of labels is 1

            model.zero_grad()
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                    translator1.zero_grad()
                    translator2.zero_grad()
                elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                    translator.zero_grad()
                else:
                    raise ValueError('Unknown modalities type')

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()

            batch_size = text.size(0)

            net = nn.DataParallel(model) if hyp_params.distribute else model
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                    trans_net1 = nn.DataParallel(translator1) if hyp_params.distribute else translator1
                    trans_net2 = nn.DataParallel(translator2) if hyp_params.distribute else translator2
                    if hyp_params.modalities == 'L':
                        fake_a = trans_net1(text, audio, 'train')
                        fake_v = trans_net2(text, vision, 'train')
                        trans_loss = trans_criterion(fake_a, audio) + trans_criterion(fake_v, vision)
                    elif hyp_params.modalities == 'A':
                        fake_l = trans_net1(audio, text, 'train')
                        fake_v = trans_net2(audio, vision, 'train')
                        trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_v, vision)
                    elif hyp_params.modalities == 'V':
                        fake_l = trans_net1(vision, text, 'train')
                        fake_a = trans_net2(vision, audio, 'train')
                        trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_a, audio)
                    else:
                        raise ValueError('Unknown modalities type')
                elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                    trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                    if hyp_params.modalities == 'LA':
                        fake_v = trans_net((text, audio), vision, 'train')
                        trans_loss = trans_criterion(fake_v, vision)
                    elif hyp_params.modalities == 'LV':
                        fake_a = trans_net((text, vision), audio, 'train')
                        trans_loss = trans_criterion(fake_a, audio)
                    elif hyp_params.modalities == 'AV':
                        fake_l = trans_net((audio, vision), text, 'train')
                        trans_loss = trans_criterion(fake_l, text)
                    else:
                        raise ValueError('Unknown modalities type')
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities == 'L':
                    preds, _ = net(text, fake_a, fake_v)
                elif hyp_params.modalities == 'A':
                    preds, _ = net(fake_l, audio, fake_v)
                elif hyp_params.modalities == 'V':
                    preds, _ = net(fake_l, fake_a, vision)
                elif hyp_params.modalities == 'LA':
                    preds, _ = net(text, audio, fake_v)
                elif hyp_params.modalities == 'LV':
                    preds, _ = net(text, fake_a, vision)
                elif hyp_params.modalities == 'AV':
                    preds, _ = net(fake_l, audio, vision)
                else:
                    raise ValueError('Unknown modalities type')
            elif hyp_params.modalities == 'LAV':
                preds, _ = net(text, audio, vision)
            else:
                raise ValueError('Unknown modalities type')
            if hyp_params.dataset == 'iemocap':
                preds = preds.view(-1, 2)
                eval_attr = eval_attr.view(-1)
            raw_loss = criterion(preds, eval_attr)
            if hyp_params.modalities != 'LAV':
                combined_loss = raw_loss + trans_loss
            else:
                combined_loss = raw_loss
            combined_loss.backward()

            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                    torch.nn.utils.clip_grad_norm_(translator1.parameters(), hyp_params.clip)
                    torch.nn.utils.clip_grad_norm_(translator2.parameters(), hyp_params.clip)
                    translator1_optimizer.step()
                    translator2_optimizer.step()
                elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                    torch.nn.utils.clip_grad_norm_(translator.parameters(), hyp_params.clip)
                    translator_optimizer.step()
                else:
                    raise ValueError('Unknown modalities type')

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(model, translator, criterion, test=False):
        if isinstance(translator, tuple):
            translator1, translator2 = translator
        model.eval()
        if hyp_params.modalities != 'LAV':
            if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                translator1.eval()
                translator2.eval()
            elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                translator.eval()
            else:
                raise ValueError('Unknown modalities type')
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()

                batch_size = text.size(0)

                net = nn.DataParallel(model) if hyp_params.distribute else model
                if hyp_params.modalities != 'LAV':
                    if not test:
                        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                            trans_net1 = nn.DataParallel(translator1) if hyp_params.distribute else translator1
                            trans_net2 = nn.DataParallel(translator2) if hyp_params.distribute else translator2
                            if hyp_params.modalities == 'L':
                                fake_a = trans_net1(text, audio, 'valid')
                                fake_v = trans_net2(text, vision, 'valid')
                                trans_loss = trans_criterion(fake_a, audio) + trans_criterion(fake_v, vision)
                            elif hyp_params.modalities == 'A':
                                fake_l = trans_net1(audio, text, 'valid')
                                fake_v = trans_net2(audio, vision, 'valid')
                                trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_v, vision)
                            elif hyp_params.modalities == 'V':
                                fake_l = trans_net1(vision, text, 'valid')
                                fake_a = trans_net2(vision, audio, 'valid')
                                trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_a, audio)
                            else:
                                raise ValueError('Unknown modalities type')
                        elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                            trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                            if hyp_params.modalities == 'LA':
                                fake_v = trans_net((text, audio), vision, 'valid')
                                trans_loss = trans_criterion(fake_v, vision)
                            elif hyp_params.modalities == 'LV':
                                fake_a = trans_net((text, vision), audio, 'valid')
                                trans_loss = trans_criterion(fake_a, audio)
                            elif hyp_params.modalities == 'AV':
                                fake_l = trans_net((audio, vision), text, 'valid')
                                trans_loss = trans_criterion(fake_l, text)
                            else:
                                raise ValueError('Unknown modalities type')
                    else:
                        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                            trans_net1 = nn.DataParallel(translator1) if hyp_params.distribute else translator1
                            trans_net2 = nn.DataParallel(translator2) if hyp_params.distribute else translator2
                            if hyp_params.modalities == 'L':
                                fake_a = torch.Tensor().cuda()
                                fake_v = torch.Tensor().cuda()
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        fake_a_token = trans_net1(text, audio, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_a_token = trans_net1(text, fake_a, 'test')[:, [-1]]
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        fake_v_token = trans_net2(text, vision, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_v_token = trans_net2(text, fake_v, 'test')[:, [-1]]
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)
                            elif hyp_params.modalities == 'A':
                                fake_l = torch.Tensor().cuda()
                                fake_v = torch.Tensor().cuda()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        fake_l_token = trans_net1(audio, text, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_l_token = trans_net1(audio, fake_l, 'test')[:, [-1]]
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        fake_v_token = trans_net2(audio, vision, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_v_token = trans_net2(audio, fake_v, 'test')[:, [-1]]
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)
                            elif hyp_params.modalities == 'V':
                                fake_l = torch.Tensor().cuda()
                                fake_a = torch.Tensor().cuda()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        fake_l_token = trans_net1(vision, text, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_l_token = trans_net1(vision, fake_l, 'test')[:, [-1]]
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        fake_a_token = trans_net2(vision, audio, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_a_token = trans_net2(vision, fake_a, 'test')[:, [-1]]
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                            else:
                                raise ValueError('Unknown modalities type')
                        elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                            trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                            if hyp_params.modalities == 'LA':
                                fake_v = torch.Tensor().cuda()
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        fake_v_token = trans_net((text, audio), vision, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_v_token = trans_net((text, audio), fake_v, 'test')[:, [-1]]
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)
                            elif hyp_params.modalities == 'LV':
                                fake_a = torch.Tensor().cuda()
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        fake_a_token = trans_net((text, vision), audio, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_a_token = trans_net((text, vision), fake_a, 'test')[:, [-1]]
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                            elif hyp_params.modalities == 'AV':
                                fake_l = torch.Tensor().cuda()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        fake_l_token = trans_net((audio, vision), text, 'test', eval_start=True)[:, [-1]]
                                    else:
                                        fake_l_token = trans_net((audio, vision), fake_l, 'test')[:, [-1]]
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)
                            else:
                                raise ValueError('Unknown modalities type')
                        else:
                            raise ValueError('Unknown modalities type')
                if hyp_params.modalities != 'LAV':
                    if hyp_params.modalities == 'L':
                        preds, _ = net(text, fake_a, fake_v)
                    elif hyp_params.modalities == 'A':
                        preds, _ = net(fake_l, audio, fake_v)
                    elif hyp_params.modalities == 'V':
                        preds, _ = net(fake_l, fake_a, vision)
                    elif hyp_params.modalities == 'LA':
                        preds, _ = net(text, audio, fake_v)
                    elif hyp_params.modalities == 'LV':
                        preds, _ = net(text, fake_a, vision)
                    elif hyp_params.modalities == 'AV':
                        preds, _ = net(fake_l, audio, vision)
                    else:
                        raise ValueError('Unknown modalities type')
                elif hyp_params.modalities == 'LAV':
                    preds, _ = net(text, audio, vision)
                else:
                    raise ValueError('Unknown modalities type')
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                raw_loss = criterion(preds, eval_attr)
                if hyp_params.modalities != 'LAV' and not test:
                    combined_loss = raw_loss + trans_loss
                else:
                    combined_loss = raw_loss
                total_loss += combined_loss.item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    if hyp_params.modalities != 'LAV':
        if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
            mgm_parameter1 = sum([param.nelement() for param in translator1.parameters()])
            mgm_parameter2 = sum([param.nelement() for param in translator2.parameters()])
            mgm_parameter = mgm_parameter1 + mgm_parameter2
        elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
            mgm_parameter = sum([param.nelement() for param in translator.parameters()])
        else:
            raise ValueError('Unknown modalities type')
        print(f'Trainable Parameters for Multimodal Generation Model (MGM): {mgm_parameter}...')
    mum_parameter = sum([param.nelement() for param in model.parameters()])
    print(f'Trainable Parameters for Multimodal Understanding Model (MUM): {mum_parameter}...')
    best_valid = 1e8
    loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)
    for epoch in loop:
    # for epoch in range(1, hyp_params.num_epochs+1):
        loop.set_description(f'Epoch {epoch:2d}/{hyp_params.num_epochs}')
        start = time.time()
        train(model, translator, optimizer, criterion)
        val_loss, _, _ = evaluate(model, translator, criterion, test=False)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)  # Decay learning rate by validation loss

        # print("-"*50)
        # print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f}'.format(epoch, duration, val_loss))
        # print("-"*50)

        if val_loss < best_valid:
            if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
                save_model(hyp_params, translator1, name='TRANSLATOR_1')
                save_model(hyp_params, translator2, name='TRANSLATOR_2')
            elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
                save_model(hyp_params, translator, name='TRANSLATOR')
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    if hyp_params.modalities == 'L' or hyp_params.modalities == 'A' or hyp_params.modalities == 'V':
        translator1 = load_model(hyp_params, name='TRANSLATOR_1')
        translator2 = load_model(hyp_params, name='TRANSLATOR_2')
        translator = (translator1, translator2)
    elif hyp_params.modalities == 'LA' or hyp_params.modalities == 'LV' or hyp_params.modalities == 'AV':
        translator = load_model(hyp_params, name='TRANSLATOR')
    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, translator, criterion, test=True)

    if hyp_params.dataset == "mosei_senti" or hyp_params.dataset == 'mosei-bert':
        acc = eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi' or hyp_params.dataset == 'mosi-bert':
        acc = eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        acc = eval_iemocap(results, truths)
    elif hyp_params.dataset == 'sims':
        acc = eval_sims(results, truths)

    return acc



# # import torch
# # from torch import nn
# # import sys
# # from src import models
# # from src.utils import *
# # import torch.optim as optim
# # import numpy as np
# # import time
# # from torch.optim.lr_scheduler import ReduceLROnPlateau
# # from tqdm import tqdm
# # from src.eval_metrics import *

# def initiate(hyp_params, train_loader, valid_loader, test_loader):
#     """
#     兼容 CFMModel (分类) + TRANSLATEModel (缺失补全) 的初始化函数
#     [Fix]: 移除了 ReduceLROnPlateau 的 verbose 参数以适配新版 PyTorch
#     """
    
#     # =====================================================================
#     # 1. 初始化生成器 (TRANSLATEModel)
#     # =====================================================================
#     translator1, translator2 = None, None
#     translator1_optimizer, translator2_optimizer = None, None
#     translator, translator_optimizer = None, None
#     trans_criterion = nn.MSELoss() # 生成任务用 MSE Loss

#     if hyp_params.modalities != 'LAV':
#         if hyp_params.modalities in ['L', 'A', 'V']:
#             missing_map = {'L': ['A', 'V'], 'A': ['L', 'V'], 'V': ['L', 'A']}
#             miss1, miss2 = missing_map[hyp_params.modalities]
            
#             translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, missing=miss1)
#             translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, missing=miss2)
            
#             if hyp_params.use_cuda:
#                 translator1, translator2 = translator1.cuda(), translator2.cuda()
                
#             translator1_optimizer = getattr(optim, hyp_params.optim)(translator1.parameters(), lr=hyp_params.lr)
#             translator2_optimizer = getattr(optim, hyp_params.optim)(translator2.parameters(), lr=hyp_params.lr)
            
#         elif hyp_params.modalities in ['LA', 'LV', 'AV']:
#             missing_map = {'LA': 'V', 'LV': 'A', 'AV': 'L'}
#             miss = missing_map[hyp_params.modalities]
            
#             translator = getattr(models, 'TRANSLATEModel')(hyp_params, missing=miss)
#             if hyp_params.use_cuda: translator = translator.cuda()
#             translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)

#     # =====================================================================
#     # 2. 初始化分类器 (CFMModel)
#     # =====================================================================
#     model = getattr(models, hyp_params.model + 'Model')(hyp_params)
#     if hyp_params.use_cuda: model = model.cuda()

#     # --- CFMModel 的优化器配置 ---
#     if hyp_params.use_bert:
#         bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#         bert_params = list(model.text_model.named_parameters())
#         bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
#         bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
#         model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]
        
#         optimizer_grouped_parameters = [
#             {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert, 'lr': hyp_params.lr_bert},
#             {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
#             {'params': model_params_other, 'weight_decay': 0.0, 'lr': hyp_params.lr}
#         ]
#         optimizer = optim.Adam(optimizer_grouped_parameters)
#     else:
#         optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)

#     criterion = getattr(nn, hyp_params.criterion)()
    
#     # [修复] 移除了 verbose=True，防止报错
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)

#     settings = {
#         'model': model,
#         'optimizer': optimizer,
#         'criterion': criterion,
#         'scheduler': scheduler,
#         'translator1': translator1, 'translator2': translator2,
#         'translator1_optimizer': translator1_optimizer, 'translator2_optimizer': translator2_optimizer,
#         'translator': translator, 'translator_optimizer': translator_optimizer,
#         'trans_criterion': trans_criterion
#     }
    
#     return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


# def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
#     model = settings['model']
#     optimizer = settings['optimizer']
#     criterion = settings['criterion']
#     scheduler = settings['scheduler']
#     trans_criterion = settings['trans_criterion']
    
#     translator1 = settings.get('translator1')
#     translator2 = settings.get('translator2')
#     translator1_opt = settings.get('translator1_optimizer')
#     translator2_opt = settings.get('translator2_optimizer')
#     translator = settings.get('translator')
#     translator_opt = settings.get('translator_optimizer')

#     def train(model, optimizer, criterion):
#         model.train()
#         if translator1: translator1.train(); translator2.train()
#         if translator: translator.train()
        
#         epoch_loss = 0
        
#         for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
#             sample_ind, text, audio, vision = batch_X
#             eval_attr = batch_Y.squeeze(-1)

#             if hyp_params.use_cuda:
#                 text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
#                 if hyp_params.dataset == 'iemocap': eval_attr = eval_attr.long()

#             batch_size = text.size(0)
            
#             model.zero_grad()
#             if translator1: translator1.zero_grad(); translator2.zero_grad()
#             if translator: translator.zero_grad()

#             # =========================================================
#             # Step 1: 生成补全
#             # =========================================================
#             trans_loss = 0.0
            
#             if hyp_params.modalities == 'L':
#                 fake_a = translator1(text, audio, 'train')
#                 fake_v = translator2(text, vision, 'train')
#                 trans_loss = trans_criterion(fake_a, audio) + trans_criterion(fake_v, vision)
#                 in_text, in_audio, in_vision = text, fake_a, fake_v
                
#             elif hyp_params.modalities == 'A':
#                 fake_l = translator1(audio, text, 'train')
#                 fake_v = translator2(audio, vision, 'train')
#                 trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_v, vision)
#                 in_text, in_audio, in_vision = fake_l, audio, fake_v
            
#             elif hyp_params.modalities == 'V':
#                 fake_l = translator1(vision, text, 'train')
#                 fake_a = translator2(vision, audio, 'train')
#                 trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_a, audio)
#                 in_text, in_audio, in_vision = fake_l, fake_a, vision
                
#             elif hyp_params.modalities == 'LA':
#                 fake_v = translator((text, audio), vision, 'train')
#                 trans_loss = trans_criterion(fake_v, vision)
#                 in_text, in_audio, in_vision = text, audio, fake_v
            
#             elif hyp_params.modalities == 'LV':
#                 fake_a = translator((text, vision), audio, 'train')
#                 trans_loss = trans_criterion(fake_a, audio)
#                 in_text, in_audio, in_vision = text, fake_a, vision
                
#             elif hyp_params.modalities == 'AV':
#                 fake_l = translator((audio, vision), text, 'train')
#                 trans_loss = trans_criterion(fake_l, text)
#                 in_text, in_audio, in_vision = fake_l, audio, vision
                
#             else:
#                 in_text, in_audio, in_vision = text, audio, vision
#                 trans_loss = 0.0

#             # =========================================================
#             # Step 2: 分类理解
#             # =========================================================
#             preds, _ = model(in_text, in_audio, in_vision)

#             if hyp_params.dataset == 'iemocap':
#                 preds = preds.view(-1, 2)
#                 eval_attr = eval_attr.view(-1)

#             raw_loss = criterion(preds, eval_attr)
            
#             combined_loss = raw_loss + trans_loss
#             combined_loss.backward()

#             if translator1:
#                 torch.nn.utils.clip_grad_norm_(translator1.parameters(), hyp_params.clip)
#                 torch.nn.utils.clip_grad_norm_(translator2.parameters(), hyp_params.clip)
#                 translator1_opt.step()
#                 translator2_opt.step()
#             if translator:
#                 torch.nn.utils.clip_grad_norm_(translator.parameters(), hyp_params.clip)
#                 translator_opt.step()
                
#             torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
#             optimizer.step()

#             epoch_loss += combined_loss.item() * batch_size

#         return epoch_loss / hyp_params.n_train

#     def evaluate(model, criterion, test=False):
#         model.eval()
#         if translator1: translator1.eval(); translator2.eval()
#         if translator: translator.eval()
        
#         loader = test_loader if test else valid_loader
#         total_loss = 0.0
#         results = []
#         truths = []

#         with torch.no_grad():
#             for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
#                 sample_ind, text, audio, vision = batch_X
#                 eval_attr = batch_Y.squeeze(dim=-1)

#                 if hyp_params.use_cuda:
#                     text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
#                     if hyp_params.dataset == 'iemocap': eval_attr = eval_attr.long()

#                 batch_size = text.size(0)
                
#                 if hyp_params.modalities == 'L':
#                     fake_a = translator1(text, audio, 'test', eval_start=True)
#                     fake_v = translator2(text, vision, 'test', eval_start=True)
#                     in_text, in_audio, in_vision = text, fake_a, fake_v
#                 elif hyp_params.modalities == 'LA':
#                     fake_v = translator((text, audio), vision, 'test', eval_start=True)
#                     in_text, in_audio, in_vision = text, audio, fake_v
#                 elif hyp_params.modalities == 'LV':
#                     fake_a = translator((text, vision), audio, 'test', eval_start=True)
#                     in_text, in_audio, in_vision = text, fake_a, vision
#                 elif hyp_params.modalities == 'AV':
#                     fake_l = translator((audio, vision), text, 'test', eval_start=True)
#                     in_text, in_audio, in_vision = fake_l, audio, vision
#                 elif hyp_params.modalities == 'A':
#                     fake_l = translator1(audio, text, 'test', eval_start=True)
#                     fake_v = translator2(audio, vision, 'test', eval_start=True)
#                     in_text, in_audio, in_vision = fake_l, audio, fake_v
#                 elif hyp_params.modalities == 'V':
#                     fake_l = translator1(vision, text, 'test', eval_start=True)
#                     fake_a = translator2(vision, audio, 'test', eval_start=True)
#                     in_text, in_audio, in_vision = fake_l, fake_a, vision
#                 else:
#                     in_text, in_audio, in_vision = text, audio, vision

#                 preds, _ = model(in_text, in_audio, in_vision)

#                 if hyp_params.dataset == 'iemocap':
#                     preds = preds.view(-1, 2)
#                     eval_attr = eval_attr.view(-1)

#                 loss = criterion(preds, eval_attr)
#                 total_loss += loss.item() * batch_size

#                 results.append(preds)
#                 truths.append(eval_attr)

#         avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
#         results = torch.cat(results)
#         truths = torch.cat(truths)
#         return avg_loss, results, truths

#     # --- Main Loop ---
#     best_valid = 1e8
#     loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)
    
#     for epoch in loop:
#         loop.set_description(f'Epoch {epoch:2d}/{hyp_params.num_epochs}')
#         train_loss = train(model, optimizer, criterion)
#         val_loss, _, _ = evaluate(model, criterion, test=False)
        
#         # [Fix] 调度器 step 不再需要 verbose 参数，上面已经去掉了
#         scheduler.step(val_loss)

#         if val_loss < best_valid:
#             if translator1:
#                 save_model(hyp_params, translator1, name='TRANSLATOR_1')
#                 save_model(hyp_params, translator2, name='TRANSLATOR_2')
#             if translator:
#                 save_model(hyp_params, translator, name='TRANSLATOR')
#             save_model(hyp_params, model, name=hyp_params.name)
#             best_valid = val_loss

#     model = load_model(hyp_params, name=hyp_params.name)
#     if translator1:
#         translator1 = load_model(hyp_params, name='TRANSLATOR_1')
#         translator2 = load_model(hyp_params, name='TRANSLATOR_2')
#     if translator:
#         translator = load_model(hyp_params, name='TRANSLATOR')
        
#     _, results, truths = evaluate(model, criterion, test=True)

#     acc = 0.0
#     if hyp_params.dataset == "mosei_senti" or hyp_params.dataset == 'mosei-bert':
#         acc = eval_mosei_senti(results, truths, True)
#     elif hyp_params.dataset == 'mosi' or hyp_params.dataset == 'mosi-bert':
#         acc = eval_mosi(results, truths, True)
#     elif hyp_params.dataset == 'iemocap':
#         acc = eval_iemocap(results, truths)
#     elif hyp_params.dataset == 'sims':
#         acc = eval_sims(results, truths)

#     return acc


# from torch.cuda.amp import autocast, GradScaler

# ####################################################################
# #
# # Construct the model
# #
# ####################################################################

# def initiate(hyp_params, train_loader, valid_loader, test_loader):
#     model = getattr(models, hyp_params.model + 'Model')(hyp_params)
#     if hyp_params.use_cuda:
#         model = model.cuda()

#     # --- 优化器配置 ---
#     if hyp_params.use_bert:
#         bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#         bert_params = list(model.text_model.named_parameters())
#         bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
#         bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
#         model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]
#         optimizer_grouped_parameters = [
#             {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert, 'lr': hyp_params.lr_bert},
#             {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
#             {'params': model_params_other, 'weight_decay': 0.0, 'lr': hyp_params.lr}
#         ]
#         optimizer = optim.Adam(optimizer_grouped_parameters)
#     else:
#         optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)

#     criterion = getattr(nn, hyp_params.criterion)()
    
#     # --- 动态构建 Translators ---
#     translators = []
#     translators_opts = []
    
#     if hyp_params.modalities != 'LAV':
#         # 定义缺失映射逻辑
#         map_gen = {
#             'L': ['A', 'V'], 'A': ['L', 'V'], 'V': ['L', 'A'],
#             'LA': ['V'], 'LV': ['A'], 'AV': ['L']
#         }
        
#         if hyp_params.modalities in map_gen:
#             targets = map_gen[hyp_params.modalities]
#             for missing_mod in targets:
#                 t_model = getattr(models, 'TRANSLATEModel')(hyp_params, missing=missing_mod)
#                 if hyp_params.use_cuda:
#                     t_model = t_model.cuda()
#                 t_opt = getattr(optim, hyp_params.optim)(t_model.parameters(), lr=hyp_params.lr)
                
#                 translators.append(t_model)
#                 translators_opts.append(t_opt)

#     scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)
    
#     settings = {
#         'model': model,
#         'optimizer': optimizer,
#         'criterion': criterion,
#         'scheduler': scheduler,
#         'translators': translators,
#         'translators_opts': translators_opts
#     }
    
#     return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


# ####################################################################
# #
# # Training and evaluation scripts
# #
# ####################################################################

# def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
#     model = settings['model']
#     optimizer = settings['optimizer']
#     criterion = settings['criterion']
#     scheduler = settings['scheduler']
#     translators = settings['translators']
#     translators_opts = settings['translators_opts']
    
#     trans_criterion = nn.MSELoss()
#     scaler = GradScaler() # 混合精度 Scaler

#     def train(model, optimizer, criterion, epoch):
#         model.train()
#         for t in translators: t.train()
        
#         epoch_total_loss = 0.0
#         epoch_cls_loss = 0.0
#         epoch_gen_loss = 0.0
#         proc_size = 0
        
#         # 内层进度条
#         pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}', leave=False)
        
#         for i_batch, (batch_X, batch_Y, batch_META) in pbar:
#             sample_ind, text, audio, vision = batch_X
#             eval_attr = batch_Y.squeeze(-1)

#             if hyp_params.use_cuda:
#                 with torch.cuda.device(0):
#                     text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
#                     if hyp_params.dataset == 'iemocap':
#                         eval_attr = eval_attr.long()

#             batch_size = text.size(0)
            
#             model.zero_grad()
#             for opt in translators_opts: opt.zero_grad()

#             # 开启混合精度上下文
#             with autocast():
#                 # ------------------------------------------------
#                 # 1. 生成阶段 (Translation)
#                 # ------------------------------------------------
#                 trans_loss = 0.0
#                 generated_features = {'L': text, 'A': audio, 'V': vision}
#                 raw_map = {'L': text, 'A': audio, 'V': vision}
                
#                 if len(translators) > 0:
#                     curr_idx = 0
#                     if hyp_params.modalities in ['L', 'A', 'V']:
#                         targets = {'L': ['A', 'V'], 'A': ['L', 'V'], 'V': ['L', 'A']}[hyp_params.modalities]
#                         src_data = raw_map[hyp_params.modalities]
                        
#                         for tgt_mod in targets:
#                             t_model = translators[curr_idx]
#                             tgt_data = raw_map[tgt_mod]
#                             # 训练时使用 Teacher Forcing (输入真实 tgt)
#                             fake_tgt = t_model(src_data, tgt_data, 'train')
#                             trans_loss += trans_criterion(fake_tgt, tgt_data)
#                             generated_features[tgt_mod] = fake_tgt 
#                             curr_idx += 1
                            
#                     elif hyp_params.modalities in ['LA', 'LV', 'AV']:
#                         target = {'LA': 'V', 'LV': 'A', 'AV': 'L'}[hyp_params.modalities]
#                         t_model = translators[0]
#                         tgt_data = raw_map[target]
#                         src_mods = list(hyp_params.modalities)
#                         src_input = (raw_map[src_mods[0]], raw_map[src_mods[1]])
                        
#                         fake_tgt = t_model(src_input, tgt_data, 'train')
#                         trans_loss += trans_criterion(fake_tgt, tgt_data)
#                         generated_features[target] = fake_tgt

#                 # ------------------------------------------------
#                 # 2. 分类阶段
#                 # ------------------------------------------------
#                 preds, _ = model(generated_features['L'], generated_features['A'], generated_features['V'])
                
#                 if hyp_params.dataset == 'iemocap':
#                     preds = preds.view(-1, 2)
#                     eval_attr = eval_attr.view(-1)
                
#                 raw_loss = criterion(preds, eval_attr)
                
#                 # ★★★ 可以在这里调整权重 (例如: raw_loss + 0.1 * trans_loss)
#                 # 建议先按 1.0 跑，如果生成 Loss 太大，再改为 0.1
#                 combined_loss = raw_loss + trans_loss

#             # ------------------------------------------------
#             # 3. 反向传播 (使用 Scaler)
#             # ------------------------------------------------
#             scaler.scale(combined_loss).backward()
            
#             for opt in translators_opts: scaler.unscale_(opt)
#             scaler.unscale_(optimizer)
            
#             # 梯度裁剪
#             for t in translators: torch.nn.utils.clip_grad_norm_(t.parameters(), hyp_params.clip)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            
#             for opt in translators_opts: scaler.step(opt)
#             scaler.step(optimizer)
#             scaler.update()

#             # ------------------------------------------------
#             # 4. 统计与日志
#             # ------------------------------------------------
#             val_total = combined_loss.item()
#             val_cls = raw_loss.item()
#             # 处理 trans_loss 可能是 tensor 或 float 0.0
#             val_gen = trans_loss.item() if isinstance(trans_loss, torch.Tensor) else trans_loss

#             epoch_total_loss += val_total * batch_size
#             epoch_cls_loss += val_cls * batch_size
#             epoch_gen_loss += val_gen * batch_size
#             proc_size += batch_size
            
#             # 实时显示在进度条右侧
#             pbar.set_postfix({
#                 'T': f"{epoch_total_loss/proc_size:.3f}", 
#                 'C': f"{epoch_cls_loss/proc_size:.3f}", 
#                 'G': f"{epoch_gen_loss/proc_size:.3f}"
#             })

#         return (epoch_total_loss / hyp_params.n_train, 
#                 epoch_cls_loss / hyp_params.n_train, 
#                 epoch_gen_loss / hyp_params.n_train)

#     def evaluate(model, criterion, test=False):
#         model.eval()
#         for t in translators: t.eval()
        
#         loader = test_loader if test else valid_loader
#         total_loss = 0.0
#         results = []
#         truths = []

#         with torch.no_grad():
#             for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
#                 sample_ind, text, audio, vision = batch_X
#                 eval_attr = batch_Y.squeeze(dim=-1)

#                 if hyp_params.use_cuda:
#                     with torch.cuda.device(0):
#                         text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
#                         if hyp_params.dataset == 'iemocap':
#                             eval_attr = eval_attr.long()

#                 batch_size = text.size(0)
#                 raw_map = {'L': text, 'A': audio, 'V': vision}
#                 generated_features = raw_map.copy()
                
#                 # --- 自回归生成逻辑 (用于验证/测试) ---
#                 if len(translators) > 0:
#                     curr_idx = 0
                    
#                     def autoregressive_generate(src_input, target_modality, translator_model):
#                         # 获取目标长度
#                         tgt_len_map = {'L': hyp_params.l_len, 'A': hyp_params.a_len, 'V': hyp_params.v_len}
#                         tgt_len = tgt_len_map[target_modality]
                        
#                         # 初始化
#                         curr_generated = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                        
#                         for i in range(tgt_len):
#                             if i == 0:
#                                 # Step 1: eval_start=True -> 使用 [Uni] token
#                                 out = translator_model(src_input, raw_map[target_modality], phase='test', eval_start=True)
#                                 token = out[:, -1:] 
#                             else:
#                                 # Step N: 输入之前生成的所有 tokens
#                                 out = translator_model(src_input, curr_generated, phase='test', eval_start=False)
#                                 token = out[:, -1:]
                            
#                             curr_generated = torch.cat([curr_generated, token], dim=1) if i > 0 else token
#                         return curr_generated

#                     if hyp_params.modalities in ['L', 'A', 'V']:
#                         targets = {'L': ['A', 'V'], 'A': ['L', 'V'], 'V': ['L', 'A']}[hyp_params.modalities]
#                         src = raw_map[hyp_params.modalities]
                        
#                         for tgt_mod in targets:
#                             # 验证集也使用严格的自回归生成，以确保 Loss 真实反映模型能力
#                             # 如果验证速度太慢，可以改为 Teacher Forcing
#                             gen_data = autoregressive_generate(src, tgt_mod, translators[curr_idx])
#                             generated_features[tgt_mod] = gen_data
#                             curr_idx += 1
                            
#                     elif hyp_params.modalities in ['LA', 'LV', 'AV']:
#                         target = {'LA': 'V', 'LV': 'A', 'AV': 'L'}[hyp_params.modalities]
#                         src_mods = list(hyp_params.modalities)
#                         src = (raw_map[src_mods[0]], raw_map[src_mods[1]])
                        
#                         gen_data = autoregressive_generate(src, target, translators[0])
#                         generated_features[target] = gen_data

#                 # --- 分类 ---
#                 preds, _ = model(generated_features['L'], generated_features['A'], generated_features['V'])

#                 if hyp_params.dataset == 'iemocap':
#                     preds = preds.view(-1, 2)
#                     eval_attr = eval_attr.view(-1)

#                 loss = criterion(preds, eval_attr)
#                 total_loss += loss.item() * batch_size

#                 results.append(preds)
#                 truths.append(eval_attr)

#         avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
#         results = torch.cat(results)
#         truths = torch.cat(truths)
#         return avg_loss, results, truths

#     # --- Main Loop ---
#     mgm_params = sum([sum(p.nelement() for p in t.parameters()) for t in translators])
#     mum_params = sum([p.nelement() for p in model.parameters()])
#     print(f'MGM Params: {mgm_params} | MUM Params: {mum_params}')

#     best_valid = 1e8
    
#     # 打印表头
#     print(f"{'Epoch':^7} | {'Time':^7} | {'Train Total':^11} | {'Train Cls':^9} | {'Train Gen':^9} | {'Valid Loss':^10}")
#     print("-" * 75)

#     # 外层使用 range，不使用 tqdm
#     for epoch in range(1, hyp_params.num_epochs + 1):
#         start = time.time()
        
#         # 接收三个返回值
#         avg_total, avg_cls, avg_gen = train(model, optimizer, criterion, epoch)
        
#         # 验证
#         val_loss, _, _ = evaluate(model, criterion, test=False) 
        
#         end = time.time()
#         duration = end - start
        
#         scheduler.step(val_loss)

#         # 打印详细日志
#         print(f"{epoch:^7d} | {duration:^7.1f} | {avg_total:^11.4f} | {avg_cls:^9.4f} | {avg_gen:^9.4f} | {val_loss:^10.4f}")

#         if val_loss < best_valid:
#             for i, t in enumerate(translators):
#                 save_model(hyp_params, t, name=f'TRANSLATOR_{i}')
#             save_model(hyp_params, model, name=hyp_params.name)
#             best_valid = val_loss

#     # --- Loading & Testing ---
#     model = load_model(hyp_params, name=hyp_params.name)
#     for i in range(len(translators)):
#         translators[i] = load_model(hyp_params, name=f'TRANSLATOR_{i}')
        
#     _, results, truths = evaluate(model, criterion, test=True)

#     acc = 0.0
#     if hyp_params.dataset == "mosei_senti" or hyp_params.dataset == 'mosei-bert':
#         acc = eval_mosei_senti(results, truths, True)
#     elif hyp_params.dataset == 'mosi' or hyp_params.dataset == 'mosi-bert':
#         acc = eval_mosi(results, truths, True)
#     elif hyp_params.dataset == 'iemocap':
#         acc = eval_iemocap(results, truths)
#     elif hyp_params.dataset == 'sims':
#         acc = eval_sims(results, truths)

#     return acc