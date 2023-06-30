import torch
from torch import nn
import torch.nn.functional as F
import einops
import random, math, sys

class SelfAttention(nn.Module):
    """
    default attention
    """
    def __init__(self, emb, heads=8):

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads

        s = emb // heads

        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)


    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        # split keys, queries and values in h different pieces
        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # scale queries and keys
        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

         
        dot = F.softmax(dot, dim=2)

        # now we have the dot products, we need to combine with the values
        out = torch.bmm(dot, values).view(b, h, t, s)
        
        # combine heads 
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)


class SelfAttentionWide(nn.Module):
    """
    full-size embedding vector for each head.
    """

    def __init__(self, emb, heads=8):

        super().__init__()

        self.emb = emb
        self.heads = heads

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, e)

        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class FastformerHead(nn.Module):
    def __init__(self, emb, head_size):
        super(FastformerHead, self).__init__()
        self.toqueries = nn.Linear(emb, head_size, bias = False)
        self.tokeys = nn.Linear(emb, head_size, bias = False)
        self.tovalues = nn.Linear(emb, head_size, bias = False)
        self.to_r = nn.Linear(head_size, head_size, bias = False) 

        self.to_query_att_logits = nn.Linear(head_size, 1, bias = False)
        self.to_key_att_logits = nn.Linear(head_size, 1, bias = False)

    def forward(self, x):
        b, t, e = x.size()
        queries = self.toqueries(x) # b, t, e
        keys = self.tokeys(x) # b, t, e
        values = self.tovalues(x) # b, t, e

        query_att_logits = self.to_query_att_logits(queries)/e**0.5 # b, t, 1, because this simulates the logits wq^T q (skipping creating w)

        alpha_att = torch.softmax(query_att_logits, dim=-1) # b, t, 1
        global_query = torch.einsum('b t e, b t e -> b e', alpha_att, queries) # b, e
        
        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, 'b e -> b copy e', copy = t)
        p = repeat_global_query * keys


        key_att_logits = self.to_key_att_logits(p)/e**0.5 # b, t, 1, because this simulates the logits wk^T p (skipping creating w)

        beta_att = torch.softmax(key_att_logits, dim=-1) # b, t, 1
        global_key = torch.einsum('b n d, b n d -> b d', beta_att, p) # b, e

        # Model the interaction between global key vector and the value vector
        repeat_global_key = einops.repeat(global_key, 'b e -> b copy e', copy = t)
        u = repeat_global_key * values

        r = self.to_r(u)

        output = r + queries
        
        return output

class FastformerMultiHead(nn.Module):
    def __init__(self, emb, head_size, heads):
        super(FastformerMultiHead, self).__init__()
        self.heads = nn.ModuleList([FastformerHead(emb, head_size) for _ in range(heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class TransformerBlock(nn.Module):
    """
    A straightforward transformer block.
    """

    def __init__(self, emb, heads, seq_length, ff_hidden_mult=4, dropout=0.0, attention_type='default',
                 ):
        super().__init__()

        if attention_type == 'default':
            self.attention = SelfAttention(emb, heads=heads)
        elif attention_type == 'wide':
            self.attention = SelfAttentionWide(emb, heads=heads)
        elif attention_type == 'fast':
            head_size = emb // heads
            self.attention = FastformerMultiHead(emb, head_size, heads=heads)
        else:
            raise Exception(f'Self-attention type {type} not recognized.')

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(

            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x
