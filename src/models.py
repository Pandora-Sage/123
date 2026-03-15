import torch
from torch import nn
import torch.nn.functional as F
import math
from modules.unimf import MultimodalTransformerEncoder, TransformerEncoder
from transformers import BertTokenizer, BertModel
from modules.position_embedding import SinusoidalPositionalEmbedding


class TRANSLATEModel(nn.Module):
    def __init__(self, hyp_params, missing=None):
        """
        Construct a Translate model.
        """
        super(TRANSLATEModel, self).__init__()
        if hyp_params.dataset == 'meld_senti' or hyp_params.dataset == 'meld_emo':
            self.l_len, self.a_len = hyp_params.l_len, hyp_params.a_len
            self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
            self.v_len, self.orig_d_v = 0, 0
        else:
            self.l_len, self.a_len, self.v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
            self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.trans_layers = hyp_params.trans_layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.trans_dropout = hyp_params.trans_dropout
        self.modalities = hyp_params.modalities  # the input modality
        self.missing = missing  # mark the missing modality

        self.position_embeddings = nn.Embedding(max(self.l_len, self.a_len, self.v_len), self.embed_dim)
        self.modal_type_embeddings = nn.Embedding(4, self.embed_dim)

        self.multi = nn.Parameter(torch.Tensor(1, self.embed_dim))
        nn.init.xavier_uniform_(self.multi)

        # translate module
        self.translator = TransformerEncoder(embed_dim=self.embed_dim,
                                             num_heads=self.num_heads,
                                             lens=(self.l_len, self.a_len, self.v_len),
                                             layers=self.trans_layers,
                                             modalities=self.modalities,
                                             missing=self.missing,
                                             attn_dropout=self.attn_dropout,
                                             relu_dropout=self.relu_dropout,
                                             res_dropout=self.res_dropout)

        # project module  # just use fc to replace conv1d :-)
        if 'L' in self.modalities or self.missing == 'L':
            self.proj_l = nn.Linear(self.orig_d_l, self.embed_dim)
        if 'A' in self.modalities or self.missing == 'A':
            self.proj_a = nn.Linear(self.orig_d_a, self.embed_dim)
        if 'V' in self.modalities or self.missing == 'V':
            self.proj_v = nn.Linear(self.orig_d_v, self.embed_dim)

        if self.missing == 'L':
            self.out = nn.Linear(self.embed_dim, self.orig_d_l)
        elif self.missing == 'A':
            self.out = nn.Linear(self.embed_dim, self.orig_d_a)
        elif self.missing == 'V':
            self.out = nn.Linear(self.embed_dim, self.orig_d_v)
        else:
            raise ValueError('Unknown missing modality type')

    def forward(self, src, tgt, phase='train', eval_start=False):
        """
        src and tgt should have dimension [batch_size, seq_len, n_features]
        """
        if self.modalities == 'L':
            if self.missing == 'A':
                x_l, x_a = src, tgt
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_l = x_l.transpose(0, 1)  # (seq, batch, embed_dim)
                x_a = x_a.transpose(0, 1)
            elif self.missing == 'V':
                x_l, x_v = src, tgt
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_l = x_l.transpose(0, 1)
                x_v = x_v.transpose(0, 1)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'A':
            if self.missing == 'L':
                x_a, x_l = src, tgt
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_a = x_a.transpose(0, 1)
                x_l = x_l.transpose(0, 1)
            elif self.missing == 'V':
                x_a, x_v = src, tgt
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_a = x_a.transpose(0, 1)
                x_v = x_v.transpose(0, 1)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'V':
            if self.missing == 'L':
                x_v, x_l = src, tgt
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_v = x_v.transpose(0, 1)
                x_l = x_l.transpose(0, 1)
            elif self.missing == 'A':
                x_v, x_a = src, tgt
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_v = x_v.transpose(0, 1)
                x_a = x_a.transpose(0, 1)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'LA':
            (x_l, x_a), x_v = src, tgt
            x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
            x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
            x_l = x_l.transpose(0, 1)
            x_a = x_a.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
        elif self.modalities == 'LV':
            (x_l, x_v), x_a = src, tgt
            x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
            x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
            x_l = x_l.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
            x_a = x_a.transpose(0, 1)
        elif self.modalities == 'AV':
            (x_a, x_v), x_l = src, tgt
            x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
            x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
            x_a = x_a.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
            x_l = x_l.transpose(0, 1)
        else:
            raise ValueError('Unknown modalities type')
        #################################################################################
        # For modal type embedding
        L_MODAL_TYPE_IDX = 0
        A_MODAL_TYPE_IDX = 1
        V_MODAL_TYPE_IDX = 2

        # Prepare the [Uni] token or [Bi] token
        # NOTE: [Uni] or [Bi] is in front of the missing modality
        batch_size = tgt.shape[0]
        multi = self.multi.unsqueeze(1).repeat(1, batch_size, 1)

        if phase != 'test':
            if self.missing == 'L':
                x_l = torch.cat((multi, x_l[:-1]), dim=0)
            elif self.missing == 'A':
                x_a = torch.cat((multi, x_a[:-1]), dim=0)
            elif self.missing == 'V':  # self.missing == 'V'
                x_v = torch.cat((multi, x_v[:-1]), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        else:
            if eval_start:
                if self.missing == 'L':
                    x_l = multi  # use [Uni] or [Bi] token as start to generate missing modality
                elif self.missing == 'A':
                    x_a = multi
                elif self.missing == 'V':
                    x_v = multi
                else:
                    raise ValueError('Unknown missing modality type')
            else:
                if self.missing == 'L':
                    x_l = torch.cat((multi, x_l), dim=0)
                elif self.missing == 'A':
                    x_a = torch.cat((multi, x_a), dim=0)
                elif self.missing == 'V':
                    x_v = torch.cat((multi, x_v), dim=0)
                else:
                    raise ValueError('Unknown missing modality type')

        # Prepare the positional embeddings & modal-type embeddings
        if 'L' in self.modalities or self.missing == 'L':
            x_l_pos_ids = torch.arange(x_l.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            l_pos_embeds = self.position_embeddings(x_l_pos_ids)
            l_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_l_pos_ids, L_MODAL_TYPE_IDX))
            l_embeds = l_pos_embeds + l_modal_type_embeds
            x_l = x_l + l_embeds
            x_l = F.dropout(x_l, p=self.embed_dropout, training=self.training)
        if 'A' in self.modalities or self.missing == 'A':
            x_a_pos_ids = torch.arange(x_a.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            a_pos_embeds = self.position_embeddings(x_a_pos_ids)
            a_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_a_pos_ids, A_MODAL_TYPE_IDX))
            a_embeds = a_pos_embeds + a_modal_type_embeds
            x_a = x_a + a_embeds
            x_a = F.dropout(x_a, p=self.embed_dropout, training=self.training)
        if 'V' in self.modalities or self.missing == 'V':
            x_v_pos_ids = torch.arange(x_v.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            v_pos_embeds = self.position_embeddings(x_v_pos_ids)
            v_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_v_pos_ids, V_MODAL_TYPE_IDX))
            v_embeds = v_pos_embeds + v_modal_type_embeds
            x_v = x_v + v_embeds
            x_v = F.dropout(x_v, p=self.embed_dropout, training=self.training)
        #################################################################################
        # Translation
        if self.modalities == 'L':
            if self.missing == 'A':
                x = torch.cat((x_l, x_a), dim=0)
            elif self.missing == 'V':
                x = torch.cat((x_l, x_v), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'A':
            if self.missing == 'L':
                x = torch.cat((x_a, x_l), dim=0)
            elif self.missing == 'V':
                x = torch.cat((x_a, x_v), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'V':
            if self.missing == 'L':
                x = torch.cat((x_v, x_l), dim=0)
            elif self.missing == 'A':
                x = torch.cat((x_v, x_a), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'LA':
            x = torch.cat((x_l, x_a, x_v), dim=0)
        elif self.modalities == 'LV':
            x = torch.cat((x_l, x_v, x_a), dim=0)
        elif self.modalities == 'AV':
            x = torch.cat((x_a, x_v, x_l), dim=0)
        else:
            raise ValueError('Unknown modalities type')

        output = self.translator(x)

        if self.modalities == 'L':
            output = output[self.l_len:].transpose(0, 1)  # (batch, seq, embed_dim)
        elif self.modalities == 'A':
            output = output[self.a_len:].transpose(0, 1)
        elif self.modalities == 'V':
            output = output[self.v_len:].transpose(0, 1)
        elif self.modalities == 'LA':
            output = output[self.l_len + self.a_len:].transpose(0, 1)
        elif self.modalities == 'LV':
            output = output[self.l_len + self.v_len:].transpose(0, 1)
        elif self.modalities == 'AV':
            output = output[self.a_len + self.v_len:].transpose(0, 1)
        else:
            raise ValueError('Unknown modalities type')

        output = self.out(output)
        return output


class UNIMFModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a UniMF model.
        """
        super(UNIMFModel, self).__init__()
        if hyp_params.dataset == 'meld_senti' or hyp_params.dataset == 'meld_emo':
            self.orig_l_len, self.orig_a_len = hyp_params.l_len, hyp_params.a_len
            self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
        else:
            self.orig_l_len, self.orig_a_len, self.orig_v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
            self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.l_kernel_size = hyp_params.l_kernel_size
        self.a_kernel_size = hyp_params.a_kernel_size
        if hyp_params.dataset != 'meld_senti' and hyp_params.dataset != 'meld_emo':
            self.v_kernel_size = hyp_params.v_kernel_size
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.multimodal_layers = hyp_params.multimodal_layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.modalities = hyp_params.modalities
        self.dataset = hyp_params.dataset
        self.language = hyp_params.language
        self.use_bert = hyp_params.use_bert

        self.distribute = hyp_params.distribute

        if self.dataset == 'meld_senti' or self.dataset == 'meld_emo':
            self.cls_len = 33
        else:
            self.cls_len = 1
        self.cls = nn.Parameter(torch.Tensor(self.cls_len, self.embed_dim))
        nn.init.xavier_uniform_(self.cls)

        # Calculate the sequence length after conv1d
        self.l_len = self.orig_l_len - self.l_kernel_size + 1
        self.a_len = self.orig_a_len - self.a_kernel_size + 1
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.v_len = self.orig_v_len - self.v_kernel_size + 1

        output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        # Prepare BERT model if use bert
        if self.use_bert:
            self.text_model = BertTextEncoder(language=hyp_params.language, use_finetune=True)

        # 1. Temporal convolutional blocks
        self.proj_l = nn.Conv1d(self.orig_d_l, self.embed_dim, kernel_size=self.l_kernel_size)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.embed_dim, kernel_size=self.a_kernel_size)
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.proj_v = nn.Conv1d(self.orig_d_v, self.embed_dim, kernel_size=self.v_kernel_size)
        if 'meld' in self.dataset:
            self.proj_cls = nn.Conv1d(self.orig_d_l + self.orig_d_a, self.embed_dim, kernel_size=1)

        # 2. GRU encoder
        self.t = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)
        self.a = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.v = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)

        # 3. Multimodal fusion block
        # 3.1. Position embeddings & Modal type embeddings
        if self.dataset == 'meld_senti' or self.dataset == 'meld_emo':
            self.position_embeddings = nn.Embedding(max(self.cls_len, self.l_len, self.a_len), self.embed_dim)
        else:
            self.position_embeddings = nn.Embedding(max(self.l_len, self.a_len, self.v_len), self.embed_dim)
        self.modal_type_embeddings = nn.Embedding(4, self.embed_dim)

        # 3.2. UniMF
        self.unimf = MultimodalTransformerEncoder(embed_dim=self.embed_dim,
                                                  num_heads=self.num_heads,
                                                  layers=self.multimodal_layers,
                                                  lens=(self.cls_len, self.l_len, self.a_len,self.v_len),
                                                  modalities=self.modalities,
                                                  attn_dropout=self.attn_dropout,
                                                  relu_dropout=self.relu_dropout,
                                                  res_dropout=self.res_dropout)

        # 4. Projection layers
        combined_dim = self.embed_dim
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def forward(self, x_l, x_a, x_v=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        if self.distribute:
            self.t.flatten_parameters()
            self.a.flatten_parameters()
            if x_v is not None:
                self.v.flatten_parameters()
        #################################################################################
        # For modal type embedding
        L_MODAL_TYPE_IDX = 0
        A_MODAL_TYPE_IDX = 1
        V_MODAL_TYPE_IDX = 2
        MULTI_MODAL_TYPE_IDX = 3

        # Prepare the [CLS] token
        batch_size = x_l.shape[0]
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            cls = self.cls.unsqueeze(1).repeat(1, batch_size, 1)
        else:
            cls = self.proj_cls(torch.cat((x_l, x_a), dim=-1).transpose(1, 2)).permute(2, 0, 1)

        # Prepare the positional embeddings & modal-type embeddings
        # NOTE: [CLS] is at the ZERO index
        cls_pos_ids = torch.arange(self.cls_len, device=x_l.device).unsqueeze(1).expand(-1, batch_size)
        h_l_pos_ids = torch.arange(self.l_len, device=x_l.device).unsqueeze(1).expand(-1, batch_size)
        h_a_pos_ids = torch.arange(self.a_len, device=x_a.device).unsqueeze(1).expand(-1, batch_size)
        if x_v is not None:
            h_v_pos_ids = torch.arange(self.v_len, device=x_v.device).unsqueeze(1).expand(-1, batch_size)

        cls_pos_embeds = self.position_embeddings(cls_pos_ids)
        h_l_pos_embeds = self.position_embeddings(h_l_pos_ids)
        h_a_pos_embeds = self.position_embeddings(h_a_pos_ids)
        if x_v is not None:
            h_v_pos_embeds = self.position_embeddings(h_v_pos_ids)

        cls_modal_type_embeds = self.modal_type_embeddings(torch.full_like(cls_pos_ids, MULTI_MODAL_TYPE_IDX))
        l_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_l_pos_ids, L_MODAL_TYPE_IDX))
        a_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_a_pos_ids, A_MODAL_TYPE_IDX))
        if x_v is not None:
            v_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_v_pos_ids, V_MODAL_TYPE_IDX))
        #################################################################################
        # Project the textual/visual/audio features & Compress the sequence length
        if self.use_bert:
            x_l = self.text_model(x_l)

        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        if x_v is not None:
            x_v = x_v.transpose(1, 2)

        proj_x_l = self.proj_l(x_l)
        proj_x_a = self.proj_a(x_a)
        if x_v is not None:
            proj_x_v = self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        if x_v is not None:
            proj_x_v = proj_x_v.permute(2, 0, 1)
        #################################################################################
        # Use GRU to encode
        h_l, _ = self.t(proj_x_l)
        h_a, _ = self.a(proj_x_a)
        if x_v is not None:
            h_v, _ = self.v(proj_x_v)
        #################################################################################
        # Add positional & modal-type embeddings
        cls_embeds = cls_pos_embeds + cls_modal_type_embeds
        l_embeds = h_l_pos_embeds + l_modal_type_embeds
        a_embeds = h_a_pos_embeds + a_modal_type_embeds
        if x_v is not None:
            v_embeds = h_v_pos_embeds + v_modal_type_embeds
        cls = cls + cls_embeds
        h_l = h_l + l_embeds
        h_a = h_a + a_embeds
        if x_v is not None:
            h_v = h_v + v_embeds
        h_l = F.dropout(h_l, p=self.embed_dropout, training=self.training)
        h_a = F.dropout(h_a, p=self.embed_dropout, training=self.training)
        if x_v is not None:
            h_v = F.dropout(h_v, p=self.embed_dropout, training=self.training)
        #################################################################################
        # Multimodal fusion
        # Get total sequence and feed into UniMF
        if x_v is not None:
            x = torch.cat((cls, h_l, h_a, h_v), dim=0)
        else:
            x = torch.cat((cls, h_l, h_a), dim=0)
        x = self.unimf(x)

        if x_v is not None:
            # A residual block
            last_hs = x[0]  # get [CLS] token for prediction
        else:
            last_hs = x[:self.cls_len]  # get [CLS] tokens for prediction

        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        if x_v is None:
            output = output.transpose(0, 1)
        return output, last_hs


class BertTextEncoder(nn.Module):
    def __init__(self, language='en', use_finetune=False):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_bert/bert_en', do_lower_case=True)
            self.model = model_class.from_pretrained('pretrained_bert/bert_en')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_bert/bert_cn')
            self.model = model_class.from_pretrained('pretrained_bert/bert_cn')

        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
    
    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states


# ... (Imports 和 TRANSLATEModel 保持不变) ...

class TextGuidedAttentionSummary(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TextGuidedAttentionSummary, self).__init__()
        # batch_first=False，因为主模型的数据流是 (Seq, Batch, Dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_cls, modal_seq, modal_mask=None):
        """
        Args:
            text_cls: (Batch, Dim) - 文本 CLS 特征，作为 Query
            modal_seq: (Seq, Batch, Dim) - 音频/视频序列，作为 Key 和 Value
            modal_mask: (Optional) padding mask
        Returns:
            summary: (Batch, Dim) - 加权后的模态摘要
        """
        # Query: 调整为 (1, Batch, Dim)
        query = text_cls.unsqueeze(0)
        
        # Key, Value: modal_seq (Seq, Batch, Dim)
        # Output: (1, Batch, Dim)
        attn_output, _ = self.attn(query, modal_seq, modal_seq, key_padding_mask=modal_mask)
        
        # 压缩回 (Batch, Dim)
        attn_output = attn_output.squeeze(0)
        
        # Residual + Norm: 保证最差情况下也有 Text 的信息，防止梯度消失
        output = self.norm(text_cls + self.dropout(attn_output))
        return output

# --- 主模型 ---
class CFMModel(nn.Module):
    def __init__(self, hyp_params):
        """
        CFMModel Optimized Version (Plan A: Text-Guided Summary)
        - Core: MLP Prompt Generator + Attention-Based Summary Conditioning
        - Change: Replaced mean(dim=0) with TextGuidedAttentionSummary for better noise filtering.
        """
        super(CFMModel, self).__init__()
        
        # --- 1. 基础参数 ---
        self.modalities = hyp_params.modalities
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.dataset = hyp_params.dataset
        self.use_bert = hyp_params.use_bert
        output_dim = hyp_params.output_dim

        # 维度计算
        if hyp_params.dataset in ['meld_senti', 'meld_emo']:
            self.orig_l_len, self.orig_a_len = hyp_params.l_len, hyp_params.a_len
            self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
        else:
            self.orig_l_len, self.orig_a_len, self.orig_v_len = (
                hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
            )
            self.orig_d_l, self.orig_d_a, self.orig_d_v = (
                hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
            )
        
        self.l_kernel_size = hyp_params.l_kernel_size
        self.a_kernel_size = hyp_params.a_kernel_size
        if hyp_params.dataset in ['meld_senti', 'meld_emo']:
            self.v_kernel_size = 1 
        else:
            self.v_kernel_size = hyp_params.v_kernel_size
        
        self.l_len = self.orig_l_len - self.l_kernel_size + 1
        self.a_len = self.orig_a_len - self.a_kernel_size + 1
        if hyp_params.dataset in ['meld_senti', 'meld_emo']:
            self.v_len = 0
        else:
            self.v_len = self.orig_v_len - self.v_kernel_size + 1

        # --- 2. 基础模块 ---
        if self.dataset in ['meld_senti', 'meld_emo']:
            self.cls_len = 33
            self.proj_cls = nn.Conv1d(self.orig_d_l + self.orig_d_a, self.embed_dim, kernel_size=1)
        else:
            self.cls_len = 1
            self.cls = nn.Parameter(torch.Tensor(self.cls_len, self.embed_dim))
            nn.init.xavier_uniform_(self.cls)
            
        if self.use_bert:
            # 假设 BertTextEncoder 在外部定义或已导入
            self.text_model = BertTextEncoder(language=getattr(hyp_params, "language", "en"), use_finetune=True)

        self.proj_l = nn.Conv1d(self.orig_d_l, self.embed_dim, kernel_size=self.l_kernel_size)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.embed_dim, kernel_size=self.a_kernel_size)
        if self.dataset not in ['meld_senti', 'meld_emo']:
            self.proj_v = nn.Conv1d(self.orig_d_v, self.embed_dim, kernel_size=self.v_kernel_size)
        
        self.position_encoding_l = SinusoidalPositionalEmbedding(self.embed_dim)
        self.position_encoding_a = SinusoidalPositionalEmbedding(self.embed_dim)
        if self.dataset not in ['meld_senti', 'meld_emo']:
            self.position_encoding_v = SinusoidalPositionalEmbedding(self.embed_dim)

        # 单模态增强器
        unimodal_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=self.num_heads, dim_feedforward=self.embed_dim * 2, 
            dropout=self.attn_dropout, norm_first=True
        )
        self.unimodal_encoder_l = nn.TransformerEncoder(unimodal_layer, num_layers=2)
        self.unimodal_encoder_a = nn.TransformerEncoder(unimodal_layer, num_layers=2)
        if self.dataset not in ['meld_senti', 'meld_emo']:
            self.unimodal_encoder_v = nn.TransformerEncoder(unimodal_layer, num_layers=2)
            
        self.norm_cls = nn.LayerNorm(self.embed_dim)

        # 归一化层
        self.ln_l = nn.LayerNorm(self.embed_dim)
        self.ln_a = nn.LayerNorm(self.embed_dim)
        if self.dataset not in ['meld_senti', 'meld_emo']:
            self.ln_v = nn.LayerNorm(self.embed_dim)

        # --- [关键配置] 3. 混合条件 Prompt 生成器 (MLP + Attention Summary) ---
        self.num_prompts = 1 
        
        # 1. Text Prompt: 由 Text CLS 生成
        self.prompt_gen_l = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.num_prompts * self.embed_dim)
        )
        
        # === [PLAN A: 初始化 Attention Summary 模块] ===
        self.summary_gen_a = TextGuidedAttentionSummary(self.embed_dim, self.num_heads, self.attn_dropout)
        
        # 2. Audio Prompt: 由 [Text CLS, Audio Summary] 生成
        self.prompt_gen_a = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim), 
            nn.GELU(),
            nn.Linear(self.embed_dim, self.num_prompts * self.embed_dim)
        )
        
        # 3. Video Prompt
        if self.dataset not in ['meld_senti', 'meld_emo']:
            # === [PLAN A: 初始化 Attention Summary 模块] ===
            self.summary_gen_v = TextGuidedAttentionSummary(self.embed_dim, self.num_heads, self.attn_dropout)
            
            self.prompt_gen_v = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.num_prompts * self.embed_dim)
            )
        
        # 初始化 Prompt Generators
        for m in self.prompt_gen_l.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
        for m in self.prompt_gen_a.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
        if self.dataset not in ['meld_senti', 'meld_emo']:
            for m in self.prompt_gen_v.modules():
                if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

        # --- 4. 深度融合 ---
        self.fusion_modal_type_embeddings = nn.Embedding(4, self.embed_dim)
        
        self.fusion_encoder = MultimodalTransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            layers=hyp_params.multimodal_layers,
            lens=(self.cls_len, self.l_len, self.a_len),
            modalities=self.modalities,
            attn_dropout=self.attn_dropout,
            embed_positions=None 
        )

        # --- 5. 输出 ---
        self.att_pool_linear = nn.Linear(self.embed_dim, 1)
        self.proj1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_layer = nn.Linear(self.embed_dim, output_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=self.embed_dim ** -0.5)

    def forward(self, x_l, x_a, x_v=None, missing_mod=None):
        """
        missing_mod: Optional parameter to manually mask out a modality for testing robustness.
        """
        if x_l is None: raise ValueError("CFMModel requires text modality.")
        batch_size = x_l.size(0)
        scale = math.sqrt(self.embed_dim)

        # --- 手动缺失实验逻辑 (Zero Masking) ---
        if missing_mod is not None:
            if 'A' in missing_mod: x_a = torch.zeros_like(x_a)
            if x_v is not None and 'V' in missing_mod: x_v = torch.zeros_like(x_v)

        if self.use_bert: x_l = self.text_model(x_l)
            
        # 1. 投影 & PosEnc
        x_l_t = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        proj_x_l = self.proj_l(x_l_t).permute(2, 0, 1) * scale
        seq_len = proj_x_l.size(0)
        batch_size = proj_x_l.size(1)
        
        pos_ids = torch.arange(seq_len, device=proj_x_l.device).unsqueeze(1).repeat(1, batch_size) 
        pe_l = self.position_encoding_l(pos_ids)
        proj_x_l = proj_x_l + pe_l

        x_a_t = F.dropout(x_a.transpose(1, 2), p=self.embed_dropout, training=self.training)
        proj_x_a = self.proj_a(x_a_t).permute(2, 0, 1) * scale
        pe_a = self.position_encoding_a(proj_x_a.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        proj_x_a = proj_x_a + pe_a
        
        proj_x_v = None
        if x_v is not None and self.dataset not in ['meld_senti', 'meld_emo']:
            x_v_t = F.dropout(x_v.transpose(1, 2), p=self.embed_dropout, training=self.training)
            proj_x_v = self.proj_v(x_v_t).permute(2, 0, 1) * scale
            pe_v = self.position_encoding_v(proj_x_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            proj_x_v = proj_x_v + pe_v

        # 2. 单模态增强
        proj_x_l = self.unimodal_encoder_l(proj_x_l)
        proj_x_a = self.unimodal_encoder_a(proj_x_a)
        if proj_x_v is not None:
            proj_x_v = self.unimodal_encoder_v(proj_x_v)

        # 3. 归一化
        proj_x_l = self.ln_l(proj_x_l)
        proj_x_a = self.ln_a(proj_x_a)
        if proj_x_v is not None:
            proj_x_v = self.ln_v(proj_x_v)

        # 4. 准备 [CLS] Token
        if self.dataset in ['meld_senti', 'meld_emo']:
            cls_input = torch.cat((x_l_t, x_a_t), dim=1)
            cls_token = self.proj_cls(cls_input).permute(2, 0, 1) * scale
            pe_cls = self.position_encoding_l(cls_token.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            cls_token = cls_token + pe_cls
            cls_feat = cls_token.mean(dim=0) 
        else:
            cls_token = self.cls.unsqueeze(1).repeat(1, batch_size, 1) * scale
            cls_feat = cls_token.squeeze(0) # (Batch, Dim)

        # === [核心逻辑] 5. 混合条件生成 Prompt (MLP + Attention Summary) ===
        
        # Text Prompt (由 CLS 生成)
        p_l = self.prompt_gen_l(cls_feat).view(batch_size, self.num_prompts, self.embed_dim).permute(1, 0, 2)
        
        # Audio Prompt (由 CLS + Attention-Weighted Audio Summary 生成)
        # [PLAN A IMPLEMENTATION]
        # 使用 Text-Guided Attention Summary
        # 输入: CLS (Batch, Dim), Audio (Seq, Batch, Dim) -> 输出: (Batch, Dim)
        summary_a = self.summary_gen_a(cls_feat, proj_x_a)
        
        cond_a = torch.cat([cls_feat, summary_a], dim=-1)
        p_a = self.prompt_gen_a(cond_a).view(batch_size, self.num_prompts, self.embed_dim).permute(1, 0, 2)
        
        # 拼接列表
        seqs_to_cat = [cls_token, p_l, proj_x_l, p_a, proj_x_a]
        
        # ID 构造
        id_list = []
        id_list.append(torch.zeros(cls_token.shape[0], dtype=torch.long, device=x_l.device)) 
        id_list.append(torch.ones(p_l.shape[0], dtype=torch.long, device=x_l.device) * 1)    
        id_list.append(torch.ones(proj_x_l.shape[0], dtype=torch.long, device=x_l.device) * 1) 
        id_list.append(torch.ones(p_a.shape[0], dtype=torch.long, device=x_l.device) * 2)    
        id_list.append(torch.ones(proj_x_a.shape[0], dtype=torch.long, device=x_l.device) * 2) 
        
        # Video Prompt (如有)
        if proj_x_v is not None and hasattr(self, 'prompt_gen_v'):
            # [PLAN A IMPLEMENTATION]
            # 使用 Text-Guided Attention Summary
            summary_v = self.summary_gen_v(cls_feat, proj_x_v)
            
            cond_v = torch.cat([cls_feat, summary_v], dim=-1)
            p_v = self.prompt_gen_v(cond_v).view(batch_size, self.num_prompts, self.embed_dim).permute(1, 0, 2)
            
            seqs_to_cat.append(p_v)
            seqs_to_cat.append(proj_x_v)
            
            id_list.append(torch.ones(p_v.shape[0], dtype=torch.long, device=x_l.device) * 3)
            id_list.append(torch.ones(proj_x_v.shape[0], dtype=torch.long, device=x_l.device) * 3)
            
        fused_input = torch.cat(seqs_to_cat, dim=0)
        
        type_ids_cat = torch.cat(id_list, dim=0)
        type_ids = type_ids_cat.unsqueeze(1).repeat(1, batch_size)
        
        fused_input = fused_input + self.fusion_modal_type_embeddings(type_ids)

        # 6. 深度融合
        final_fused = self.fusion_encoder(fused_input)

        # 7. 预测
        if self.dataset in ['meld_senti', 'meld_emo']:
             last_hs = final_fused[:cls_token.shape[0]]
        else:
            att_scores = self.att_pool_linear(final_fused) 
            att_weights = F.softmax(att_scores, dim=0)     
            last_hs = (att_weights * final_fused).sum(dim=0)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        
        if self.dataset in ['meld_senti', 'meld_emo']:
            output = output.transpose(0, 1)
            
        return output, last_hs