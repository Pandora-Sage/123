import torch
from torch import nn
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
import math
from torch.nn import Parameter
from modules.masked_multihead_attention import MultiheadAttention
import sys

# ================ transformer.py

# class TransformerEncoder(nn.Module):
#     """
#     Transformer encoder consisting of *args.encoder_layers* layers. Each layer
#     is a :class:`TransformerEncoderLayer`.
#     Args:
#         embed_tokens (torch.nn.Embedding): input embedding
#         num_heads (int): number of heads
#         layers (int): number of layers
#         attn_dropout (float): dropout applied on the attention weights
#         relu_dropout (float): dropout applied on the first layer of the residual block
#         res_dropout (float): dropout applied on the residual block
#         attn_mask (bool): whether to apply mask on the attention weights
#     """

#     def __init__(self, embed_dim, num_heads, layers, lens, modalities, missing, attn_dropout=0.0, relu_dropout=0.0,
#                  res_dropout=0.0, embed_dropout=0.0, attn_mask=False, embed_positions=None):
#         super().__init__()
#         self.dropout = embed_dropout      # Embedding dropout
#         self.attn_dropout = attn_dropout
#         self.embed_dim = embed_dim

#         if embed_positions is not None:
#             self.embed_scale = math.sqrt(embed_dim)
#             self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
#         else:
#             self.embed_scale = 1
#             self.embed_positions = embed_positions
        
#         self.attn_mask = attn_mask

#         self.layers = nn.ModuleList([])
#         for layer in range(layers):
#             new_layer = TransformerEncoderLayer(embed_dim,
#                                                 lens=lens,
#                                                 modalities=modalities,
#                                                 missing=missing,
#                                                 num_heads=num_heads,
#                                                 attn_dropout=attn_dropout,
#                                                 relu_dropout=relu_dropout,
#                                                 res_dropout=res_dropout,
#                                                 attn_mask=attn_mask)
#             self.layers.append(new_layer)

#         self.register_buffer('version', torch.Tensor([2]))
#         self.normalize = True
#         if self.normalize:
#             self.layer_norm = LayerNorm(embed_dim)

#     def forward(self, x_in, x_in_k = None, x_in_v = None):
#         """
#         Args:
#             x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
#             x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
#             x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
#         Returns:
#             dict:
#                 - **encoder_out** (Tensor): the last encoder layer's output of
#                   shape `(src_len, batch, embed_dim)`
#                 - **encoder_padding_mask** (ByteTensor): the positions of
#                   padding elements of shape `(batch, src_len)`
#         """
#         # embed tokens and positions
#         x = self.embed_scale * x_in
#         if self.embed_positions is not None:
#             x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding

#         if x_in_k is not None and x_in_v is not None:
#             # embed tokens and positions    
#             x_k = self.embed_scale * x_in_k
#             x_v = self.embed_scale * x_in_v
#             if self.embed_positions is not None:
#                 x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
#                 x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        
#         # encoder layers
#         intermediates = [x]
#         for layer in self.layers:
#             if x_in_k is not None and x_in_v is not None:
#                 x = layer(x, x_k, x_v)
#             else:
#                 x = layer(x)
#             intermediates.append(x)

#         if self.normalize:
#             x = self.layer_norm(x)

#         return x

#     def max_positions(self):
#         """Maximum input length supported by the encoder."""
#         if self.embed_positions is None:
#             return self.max_source_positions
#         return min(self.max_source_positions, self.embed_positions.max_positions())


# class TransformerEncoderLayer(nn.Module):
#     """Encoder layer block.
#     In the original paper each operation (multi-head attention or FFN) is
#     postprocessed with: `dropout -> add residual -> layernorm`. In the
#     tensor2tensor code they suggest that learning is more robust when
#     preprocessing each layer with layernorm and postprocessing with:
#     `dropout -> add residual`. We default to the approach in the paper, but the
#     tensor2tensor approach can be enabled by setting
#     *args.encoder_normalize_before* to ``True``.
#     Args:
#         embed_dim: Embedding dimension
#     """

#     def __init__(self, embed_dim, lens, modalities, missing, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
#                  attn_mask=False):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
        
#         self.self_attn = MultiheadAttention(
#             embed_dim=self.embed_dim,
#             num_heads=self.num_heads,
#             lens=lens,
#             modalities=modalities,
#             missing=missing,
#             attn_dropout=attn_dropout
#         )
#         self.attn_mask = attn_mask

#         self.relu_dropout = relu_dropout
#         self.res_dropout = res_dropout
#         self.normalize_before = True

#         self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
#         self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
#         self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

#     def forward(self, x, x_k=None, x_v=None):
#         """
#         Args:
#             x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
#             encoder_padding_mask (ByteTensor): binary ByteTensor of shape
#                 `(batch, src_len)` where padding elements are indicated by ``1``.
#             x_k (Tensor): same as x
#             x_v (Tensor): same as x
#         Returns:
#             encoded output of shape `(batch, src_len, embed_dim)`
#         """
#         residual = x
#         x = self.maybe_layer_norm(0, x, before=True)
#         mask = buffered_future_mask(x, x_k) if self.attn_mask else None
#         if x_k is None and x_v is None:
#             x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
#         else:
#             x_k = self.maybe_layer_norm(0, x_k, before=True)
#             x_v = self.maybe_layer_norm(0, x_v, before=True) 
#             x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
#         x = F.dropout(x, p=self.res_dropout, training=self.training)
#         x = residual + x
#         x = self.maybe_layer_norm(0, x, after=True)

#         residual = x
#         x = self.maybe_layer_norm(1, x, before=True)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, p=self.relu_dropout, training=self.training)
#         x = self.fc2(x)
#         x = F.dropout(x, p=self.res_dropout, training=self.training)
#         x = residual + x
#         x = self.maybe_layer_norm(1, x, after=True)
#         return x

#     def maybe_layer_norm(self, i, x, before=False, after=False):
#         assert before ^ after
#         if after ^ self.normalize_before:
#             return self.layer_norms[i](x)
#         else:
#             return x

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


# if __name__ == '__main__':
#     encoder = TransformerEncoder(300, 4, 2)
#     x = torch.tensor(torch.rand(20, 2, 300))
#     print(encoder(x).shape)

# ===================

# class RotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
#         super().__init__()
#         self.dim = dim
#         self.base = base
#         # 缓存频率，避免每次 forward 都重新计算
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
#         self.register_buffer("inv_freq", inv_freq)
#         self.max_seq_len_cached = max_position_embeddings
#         self._update_cos_sin_tables(max_position_embeddings, device)

#     def _update_cos_sin_tables(self, x_len, device):
#         self.max_seq_len_cached = x_len
#         t = torch.arange(x_len, device=device, dtype=self.inv_freq.dtype)
#         freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#         # 拼接成 [cos, cos, ..., sin, sin, ...] 的形式以便后续旋转
#         emb = torch.cat((freqs, freqs), dim=-1)
#         # 缓存为 [1, seq_len, 1, dim] 格式，方便 broadcasting
#         self.register_buffer("cos_cached", emb.cos()[None, :, :], persistent=False)
#         self.register_buffer("sin_cached", emb.sin()[None, :, :], persistent=False)

#     def forward(self, x, seq_len=None):
#         # x: [batch, seq_len, dim]
#         if seq_len > self.max_seq_len_cached:
#             self._update_cos_sin_tables(seq_len, x.device)
            
#         return (
#             self.cos_cached[:, :seq_len, ...].to(dtype=x.dtype),
#             self.sin_cached[:, :seq_len, ...].to(dtype=x.dtype),
#         )

# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)

# def apply_rotary_pos_emb(q, k, cos, sin):
#     """
#     q, k: [batch, seq_len, head_dim]
#     cos, sin: [1, seq_len, head_dim]
#     """
#     # 这里的 cos 和 sin 会自动广播到 batch 维度
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# =========multihead_attention.py
# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, lens, modalities, missing=None, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.l_len, self.a_len, self.v_len = lens
        self.modalities = modalities
        self.missing = missing
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        #################################################################################
        # Masked Multi-Head Self-Attention
        # mask = torch.zeros_like(attn_weights)
        # if self.modalities == 'L':
        #     mask[:, :self.l_len, self.l_len + 1:] = float('-inf')
        #     mask[:, self.l_len + 1:, :self.l_len] = float('-inf')
        #     if self.missing == 'A':
        #         a_len = src_len - self.l_len
        #         a_mask = self.generate_square_subsequent_mask(a_len)
        #         mask[:, self.l_len:, self.l_len:] = a_mask
        #     elif self.missing == 'V':
        #         v_len = src_len - self.l_len
        #         v_mask = self.generate_square_subsequent_mask(v_len)
        #         mask[:, self.l_len:, self.l_len:] = v_mask
        #     else:
        #         raise ValueError('Unknown missing modality type')
        # elif self.modalities == 'A':
        #     mask[:, :self.a_len, self.a_len + 1:] = float('-inf')
        #     mask[:, self.a_len + 1:, :self.a_len] = float('-inf')
        #     if self.missing == 'L':
        #         l_len = src_len - self.a_len
        #         l_mask = self.generate_square_subsequent_mask(l_len)
        #         mask[:, self.a_len:, self.a_len:] = l_mask
        #     elif self.missing == 'V':
        #         v_len = src_len - self.a_len
        #         v_mask = self.generate_square_subsequent_mask(v_len)
        #         mask[:, self.a_len:, self.a_len:] = v_mask
        #     else:
        #         raise ValueError('Unknown missing modality type')
        # elif self.modalities == 'V':
        #     mask[:, :self.v_len, self.v_len + 1:] = float('-inf')
        #     mask[:, self.v_len + 1:, :self.v_len] = float('-inf')
        #     if self.missing == 'L':
        #         l_len = src_len - self.v_len
        #         l_mask = self.generate_square_subsequent_mask(l_len)
        #         mask[:, self.v_len:, self.v_len:] = l_mask
        #     elif self.missing == 'A':
        #         a_len = src_len - self.v_len
        #         a_mask = self.generate_square_subsequent_mask(a_len)
        #         mask[:, self.v_len:, self.v_len:] = a_mask
        #     else:
        #         raise ValueError('Unknown missing modality type')
        # elif self.modalities == 'LA':
        #     v_len = src_len - self.l_len - self.a_len
        #     mask[:, :self.l_len, self.l_len:self.l_len + self.a_len] = float('-inf')
        #     mask[:, self.l_len:self.l_len + self.a_len, :self.l_len] = float('-inf')
        #     mask[:, :self.l_len + self.a_len, self.l_len + self.a_len + 1:] = float('-inf')
        #     mask[:, self.l_len + self.a_len + 1:, :self.l_len + self.a_len] = float('-inf')
        #     v_mask = self.generate_square_subsequent_mask(v_len)
        #     mask[:, self.l_len + self.a_len:, self.l_len + self.a_len:] = v_mask
        # elif self.modalities == 'LV':
        #     a_len = src_len - self.l_len - self.v_len
        #     mask[:, :self.l_len, self.l_len:self.l_len + self.v_len] = float('-inf')
        #     mask[:, self.l_len:self.l_len + self.v_len, :self.l_len] = float('-inf')
        #     mask[:, :self.l_len + self.v_len, self.l_len + self.v_len + 1:] = float('-inf')
        #     mask[:, self.l_len + self.v_len + 1:, :self.l_len + self.v_len] = float('-inf')
        #     a_mask = self.generate_square_subsequent_mask(a_len)
        #     mask[:, self.l_len + self.v_len:, self.l_len + self.v_len:] = a_mask
        # elif self.modalities == 'AV':
        #     l_len = src_len - self.a_len - self.v_len
        #     mask[:, :self.a_len, self.a_len:self.a_len + self.v_len] = float('-inf')
        #     mask[:, self.a_len:self.a_len + self.v_len, :self.a_len] = float('-inf')
        #     mask[:, :self.a_len + self.v_len, self.a_len + self.v_len + 1:] = float('-inf')
        #     mask[:, self.a_len + self.v_len + 1:, :self.a_len + self.v_len] = float('-inf')
        #     l_mask = self.generate_square_subsequent_mask(l_len)
        #     mask[:, self.a_len + self.v_len:, self.a_len + self.v_len:] = l_mask
        # else:
        #     raise ValueError('Unknown modalities type')
        # attn_weights = attn_weights + mask
        #################################################################################
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



class MultimodalTransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, lens, modalities, attn_dropout=0.0, relu_dropout=0.0,
                 res_dropout=0.0, embed_dropout=0.0, attn_mask=False, embed_positions=None):
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim

        if embed_positions is not None:
            self.embed_scale = math.sqrt(embed_dim)
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        else:
            self.embed_scale = 1
            self.embed_positions = embed_positions

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = MultimodalTransformerEncoderLayer(embed_dim,
                                                          lens=lens,
                                                          modalities=modalities,
                                                          num_heads=num_heads,
                                                          attn_dropout=attn_dropout,
                                                          relu_dropout=relu_dropout,
                                                          res_dropout=res_dropout,
                                                          attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions    
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class MultimodalTransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, lens, modalities, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            lens=lens,
            modalities=modalities,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)  # The "Add & Norm" part in the paper
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


if __name__ == '__main__':
    encoder = MultimodalTransformerEncoder(300, 4, 2)
    x = torch.tensor(torch.rand(20, 2, 300))
    print(encoder(x).shape)
    


     