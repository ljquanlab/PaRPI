import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, source_dim, target_dim, hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.W_Q = nn.Linear(target_dim, num_heads * target_dim, bias=qkv_bias)
        self.W_K = nn.Linear(source_dim, num_heads * source_dim, bias=qkv_bias)
        self.W_V = nn.Linear(source_dim, num_heads * source_dim, bias=qkv_bias)

        self.W_att = nn.Parameter(torch.Tensor(num_heads, hidden_dim, hidden_dim))

        nn.init.xavier_uniform_(self.W_att)  

        self.scale = hidden_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.W_O = nn.Linear(num_heads * target_dim, target_dim, bias=qkv_bias)

        self.norm = nn.LayerNorm(target_dim)

        self.proj_drop = nn.Dropout(proj_drop)  

    def forward(self, target, source, mask=None):

        B, N, C = target.shape  
        P = source.shape[1]  
        heads = self.num_heads

        residual = target
        target = self.norm(target)  


        target_Q = self.W_Q(target).reshape(B, N, heads, C).permute(0, 2, 1, 3)  
        source_K = self.W_K(source).reshape(B, P, heads, C).permute(0, 2, 1, 3)  
        source_V = self.W_V(source).reshape(B, P, heads, C).permute(0, 2, 1, 3) 
        # calulate the attention
        source_K = torch.matmul(source_K, self.W_att) / math.sqrt(C) 
        attn = torch.matmul(target_Q, source_K.transpose(-1, -2)) / math.sqrt(C) 
        if mask != None:
            attn = attn.masked_fill(mask == 0, float("-inf"))  
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)  
        target = torch.matmul(attn, source_V).permute(0, 2, 1, 3).reshape(B, N, -1)  
        # 聚合heads
        target = self.W_O(F.elu(target.view(B, N, -1)))  
        return residual + target


