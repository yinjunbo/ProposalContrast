import torch
from torch import nn, einsum
from einops import repeat
import torch.nn.functional as F

def exists(val):
    return val is not None

def max_value(t):
    return torch.finfo(t.dtype).max

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


class ProposalEncodingLayerV2(nn.Module):
    def __init__(
        self,
        *,
        dim,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4,
        downsample = 4,
    ):
        super().__init__()

        self.inter_channels = dim // downsample

        self.g = nn.Linear(dim, self.inter_channels, bias=False)
        self.theta = nn.Linear(dim, self.inter_channels, bias=False)
        self.phi = nn.Linear(dim, self.inter_channels, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, self.inter_channels)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(self.inter_channels, self.inter_channels * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(self.inter_channels * attn_mlp_hidden_mult, self.inter_channels),
        )

        self.conv_out = nn.Linear(self.inter_channels, dim, bias=False)


    def forward(self, x, pos, mode='cross', mask = None):

        x, y = x[0], x[1]
        x_pos, y_pos = pos

        v = self.g(y)
        q = self.theta(x)
        k = self.phi(y)

        # calculate relative positional embeddings
        rel_pos = x_pos[:, :, None, :] - y_pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None, :] - k[:, None, :, :]

        # prepare mask
        if exists(mask):
            mask = mask[:, :, None] * mask[:, None, :]

        # expand values
        v = repeat(v, 'b j d -> b i j d', i=1)

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # masking
        if exists(mask):
            mask_value = -max_value(sim)
            sim.masked_fill_(~mask[..., None], mask_value)

        # attention
        attn = sim.softmax(dim=-2)

        # aggregate
        agg = einsum('b i j d, b i j d -> b i d', attn, v)

        return x + self.conv_out(agg)


class ProposalEncodingLayerV1(nn.Module):
    def __init__(
            self,
            input_dim,
            pos_dim,
            downsample,
            num_heads,
            dropout,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = input_dim//downsample
        self.head_dim = self.embed_dim // num_heads
        self.dropout_p = dropout
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # point transformer
        self.g = nn.Linear(input_dim+self.embed_dim, self.embed_dim, bias=False)
        self.theta = nn.Linear(input_dim+self.embed_dim, self.embed_dim, bias=False)
        self.phi = nn.Linear(input_dim+self.embed_dim, self.embed_dim, bias=False)

        self.query_pos_mlp = nn.Sequential(
            nn.Linear(3, pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, self.embed_dim)
        )
        self.key_pos_mlp = nn.Sequential(
            nn.Linear(3, pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, self.embed_dim)
        )

        self.out = nn.Linear(self.embed_dim, input_dim, bias=False)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x, pos, mode='cross'):

        x, y = x

        x_pos, y_pos = pos

        query_pos_emb = self.query_pos_mlp(x_pos)
        rel_pos = x_pos[:, :, None, :] - y_pos[:, None, :, :]
        key_pos_emb = self.key_pos_mlp(rel_pos)

        message = x[:, :, None, :] - y[:, None, :, :]

        q = self.theta(torch.cat([x, query_pos_emb], dim=-1)).transpose(0, 1)
        k = self.phi(torch.cat([message, key_pos_emb], dim=-1)).squeeze().transpose(0, 1)
        v = self.g(torch.cat([message, key_pos_emb], dim=-1)).squeeze().transpose(0, 1)

        bsz, tgt_len = x.size()[:2]

        head_dim = self.embed_dim // self.num_heads
        assert head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q = q * scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout_p, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn_output = self.out(attn_output.transpose(0, 1))

        attn_output = x + self.dropout(attn_output)
        attn_output = self.norm(attn_output)

        return attn_output

