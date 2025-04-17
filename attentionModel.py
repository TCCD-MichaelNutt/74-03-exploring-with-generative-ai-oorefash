'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY SR
Description: attentionModel
FilePath: /compression/attentionModel.py
'''
import torch
import torch.nn as nn
import math
import copy
from networkTool import device

# Self-attention module with multiple heads
class SelfMultiheadAttention(nn.Module):

    def __init__(self, emsize, nhead, dropout=0.5):
        super(SelfMultiheadAttention, self).__init__()
        self.nhead = nhead  # Number of attention heads
        self.head_size = emsize // nhead  # Size of each head
        assert self.head_size * nhead == emsize, "embed_dim must be divisible by num_heads"

        self.all_head_size = int(self.nhead * self.head_size)  # Total size after combining all heads
        self.mlpKey = nn.Linear(emsize, self.all_head_size)    # Linear projection for keys
        self.mlpQuery = nn.Linear(emsize, self.all_head_size)  # Linear projection for queries
        self.mlpValue = nn.Linear(emsize, self.all_head_size)  # Linear projection for values
        self.dropout = nn.Dropout(dropout)

    # Slice the projections for multi-head attention
    def slice(self,x,dim):
        new_x_shape = x.size()[:-1] + (self.nhead, self.head_size)
        x = x.view(*new_x_shape)
        if (dim == 3):
            x = x.permute(0, 2, 1, 3)  # Permute for attention computation
        elif (dim == 4):
            x = x.permute(0,1,3,2,4)
            assert 0  # This code path should not be reached
        return x

    # em: [seq_len, batch_size, embed_dim], mask: [seq_len, seq_len]
    def forward(self,em,mask):
        em = em.transpose(0,1).contiguous()  # Change to [batch_size, seq_len, embed_dim]
        Key = self.slice(self.mlpKey(em),em.dim())
        Query = self.slice(self.mlpQuery(em),em.dim())
        Value = self.slice(self.mlpValue(em),em.dim())

        # Compute scaled dot-product attention scores
        attention_score = torch.matmul(Query, Key.transpose(-1, -2)) / math.sqrt(self.head_size)
        attention_score = attention_score + mask  # Apply mask for causal/self-attention

        # Apply softmax and dropout to get attention weights
        attention_map = self.dropout(nn.Softmax(dim=-1)(attention_score))

        # Weighted sum of values
        context = torch.matmul(attention_map, Value)

        # Rearrange context tensor back to original format
        if (context.dim() == 4):
            context = context.permute(0, 2, 1, 3).contiguous()
        elif (context.dim()==5):
            context = context.permute(0, 1, 3, 2, 4).contiguous()

        # Merge multiple heads
        context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*context_shape)
        context = context.transpose(0,1).contiguous()  # Change back to [seq_len, batch_size, embed_dim]
        return context

# Single transformer encoder block
class TransformerLayer(nn.Module):

    def __init__(self, ninp, nhead, nhid, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.MultiAttention = SelfMultiheadAttention(emsize=ninp,nhead=nhead)  # Multi-head attention module
        self.linear1 = nn.Linear(ninp,nhid)  # First linear layer in feed-forward network
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(nhid,ninp)  # Second linear layer

        # Layer normalization
        self.norm1 = nn.LayerNorm(ninp, eps=1e-5)
        self.norm2 = nn.LayerNorm(ninp, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # src: input tensor, src_mask: attention mask
    def forward(self, src, src_mask):
        src2 = self.MultiAttention(src,src_mask)  # Apply attention
        src = self.dropout1(src2) + src  # Add & dropout
        src = self.norm1(src)  # Normalize
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))  # Feed-forward network
        src = src + self.dropout2(src2)  # Add & dropout
        src = self.norm2(src)  # Normalize again
        return src

# Transformer model consisting of multiple layers
class TransformerModule(nn.Module):

    def __init__(self,layer, nlayers):
        super(TransformerModule, self).__init__()
        # Deep copy the same transformer layer 'nlayers' times
        self.layers = torch.nn.ModuleList([copy.deepcopy(layer) for i in range(nlayers)])

    def forward(self,src,src_mask):
        output = src

        # Pass the input through each transformer layer
        for mod in self.layers:
            output = mod(output, src_mask=src_mask)
        return output
