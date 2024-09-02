import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=150):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Attention(nn.Module):
    def __init__(self, dim_emb, num_heads=8, qkv_bias=False, attn_do_rate=0., proj_do_rate=0.):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        dim_each_head = dim_emb // num_heads
        self.scale = dim_each_head ** -0.5

        self.qkv = nn.Linear(dim_emb, dim_emb * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_do_rate)
        self.proj = nn.Linear(dim_emb, dim_emb)  
        self.proj_dropout = nn.Dropout(proj_do_rate)

    def forward(self, x, mask=None):

        B, N, C = x.shape  

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class Attention_tm(nn.Module):
    def __init__(self, dim_emb, num_heads=8, qkv_bias=False, attn_do_rate=0., proj_do_rate=0.):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        dim_each_head = dim_emb // num_heads
        self.scale = dim_each_head ** -0.5

        self.q = nn.Linear(dim_emb, dim_emb, bias=qkv_bias)
        self.kv = nn.Linear(dim_emb, dim_emb * 2, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_do_rate)
        self.proj = nn.Linear(dim_emb, dim_emb)  
        self.proj_dropout = nn.Dropout(proj_do_rate)

    def forward(self, x, mask=None, tmp_mask=None, x_avg_lastblock=None):
        
        x_nan = x.clone()
        tmp_mask_extend = tmp_mask.unsqueeze(2).repeat(1,1,x_nan.shape[2])
        x_nan[tmp_mask_extend==False] = torch.nan
        x_avg = torch.nanmean(x_nan, dim=1).unsqueeze(1)

        if x_avg_lastblock is not None:
            x_avg = x_avg*0.7 + x_avg_lastblock*0.3
        
        B, N, C = x.shape  # b, f, j*c

        q = self.q(x)
        q = q.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]

        kv = self.kv(torch.cat((x_avg, x), dim=1))
        kv = kv.reshape(B, N+1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        mask_ = torch.ones((mask.shape[0], mask.shape[1], mask.shape[2],1)).cuda()
        mask = torch.cat((mask_, mask), dim=3)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_dropout(x)

        x_nan = x.clone()
        x_nan[tmp_mask_extend==False] = torch.nan
        x_avg = torch.nanmean(x_nan, dim=1).unsqueeze(1)
        
        return x, x_avg

class Attention_cross(nn.Module):
    def __init__(self, dim_emb, num_heads=8, qkv_bias=False, attn_do_rate=0., proj_do_rate=0.):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        dim_each_head = dim_emb // num_heads
        self.scale = dim_each_head ** -0.5

        self.q = nn.Linear(dim_emb, dim_emb, bias=qkv_bias)
        self.kv = nn.Linear(dim_emb, dim_emb * 2, bias=qkv_bias)
        
        self.attn_dropout = nn.Dropout(attn_do_rate)
        self.proj = nn.Linear(dim_emb, dim_emb)  
        self.proj_dropout = nn.Dropout(proj_do_rate)

    def forward(self, x, x_t1, x_t2, mask=None):

        B, N, C = x.shape

        q = self.q(x)
        x_cat = torch.cat((x, x_t1, x_t2), axis=1)
        kv = self.kv(x_cat)

        q = q.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        num_token = N*3
        kv = kv.reshape(B, num_token, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, do_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(do_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_joint=50, num_frame=150, num_p=12, dim_emb=48, 
                num_heads=8, ff_expand=1.0, qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0., positional_emb_type='learnalbe', cross_rolls=[1,3]):

        super(TransformerEncoder, self).__init__()

        self.num_frame = num_frame 
        self.positional_emb_type = positional_emb_type
        self.num_p = num_p
        self.joint_per_p = int(num_joint/self.num_p)

        # for learnable positional embedding
        self.positional_emb = nn.Parameter(torch.zeros(1, num_frame, num_joint, dim_emb))

        # for fixed positional embedding (ablation)
        self.tm_pos_encoder = PositionalEmbedding(num_joint*dim_emb, num_frame)
        self.sp_pos_encoder = PositionalEmbedding(dim_emb, num_joint)

        self.norm1_sp = nn.LayerNorm(dim_emb)
        self.norm1_p2p = nn.LayerNorm(dim_emb*self.joint_per_p)
        self.norm1_tm = nn.LayerNorm(dim_emb*num_joint)

        self.attention_sp = Attention(dim_emb, num_heads, qkv_bias, attn_do_rate, proj_do_rate)
        self.attention_p2p = Attention_cross(dim_emb*self.joint_per_p, num_heads, qkv_bias, attn_do_rate, proj_do_rate)
        self.attention_tm = Attention_tm(dim_emb*num_joint, num_heads, qkv_bias, 0.0, proj_do_rate)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim_emb*num_joint)
        self.feedforward = FeedForward(in_features=dim_emb*num_joint, hidden_features=int(dim_emb*num_joint*ff_expand), 
                                        out_features=dim_emb*num_joint, do_rate=proj_do_rate)
        self.mlp_avg = FeedForward(in_features=dim_emb*num_joint, hidden_features=int(dim_emb*num_joint*ff_expand), 
                                        out_features=dim_emb*num_joint, do_rate=proj_do_rate)
                            
        self.roll1 = cross_rolls[0]
        self.roll2 = cross_rolls[1]


    def forward(self, x, mask=None, positional_emb=False, tmp_mask=None, x_avg_lastblock=None):

        b, f, j, c = x.shape

        ## spatial-MHA
        x_sp = rearrange(x, 'b f j c  -> (b f) j c', )
        if positional_emb==True:
            if self.positional_emb_type=='fix':
                x_sp = x_sp + self.sp_pos_encoder(x_sp)
            else:
                pos_emb = self.positional_emb.repeat(b, 1,1,1)
                pos_emb = rearrange(pos_emb, 'b f j c -> (b f) j c', b=b, f=f)
                
                x_sp = x_sp + pos_emb

        x_sp = x_sp + self.drop_path(self.attention_sp(self.norm1_sp(x_sp), mask=None))
  
        ## person to person-MHA
        x_per = rearrange(x_sp, '(b f) (j p) c -> b f p (j c)', b=b, f=f, j=self.joint_per_p, p=self.num_p)
        norm_x_per = self.norm1_p2p(x_per)
        roll_norm_x_per_1 = torch.roll(norm_x_per, -self.roll1, dims=1)
        roll_norm_x_per_2 = torch.roll(norm_x_per, -self.roll2, dims=1)

        x_per = rearrange(x_per, 'b f p (j c)-> (b f) p (j c)', b=b, f=f, j=self.joint_per_p, p=self.num_p)
        norm_x_per = rearrange(norm_x_per, 'b f p (j c)-> (b f) p (j c)', b=b, f=f, j=self.joint_per_p, p=self.num_p)
        roll_norm_x_per_1 = rearrange(roll_norm_x_per_1, 'b f p (j c)-> (b f) p (j c)', b=b, f=f, j=self.joint_per_p, p=self.num_p)
        roll_norm_x_per_2 = rearrange(roll_norm_x_per_2, 'b f p (j c)-> (b f) p (j c)', b=b, f=f, j=self.joint_per_p, p=self.num_p)

        x_per = x_per + self.drop_path(self.attention_p2p(norm_x_per, roll_norm_x_per_1, roll_norm_x_per_2, mask=None))
  
        ## temporal-MHA
        x_tm = rearrange(x_per, '(b f) p (j c) -> b f (j p c)', b=b, f=f,  j=self.joint_per_p, p=self.num_p)
        if positional_emb==True:
            if self.positional_emb_type=='fix':
                x_tm = x_tm + self.tm_pos_encoder(x_tm)
            else:
                pos_emb = rearrange(pos_emb, '(b f) j c -> b f (j c)', b=b, f=f)
                x_tm = x_tm + pos_emb

        x, x_avg = self.attention_tm(self.norm1_tm(x_tm), mask=mask, tmp_mask=tmp_mask, x_avg_lastblock=x_avg_lastblock)
        x_tm = x_tm + self.drop_path(x)
        
        x_out = x_tm
        x_out = x_out + self.drop_path(self.feedforward(self.norm2(x_out)))
        x_out = rearrange(x_out, 'b f (j c)  -> b f j c', j=j)

        x_avg = self.mlp_avg(x_avg)
        return x_out, x_avg

class ST_Transformer(nn.Module):

    def __init__(self, num_frame, num_joint, num_p, input_channel, dim_joint_emb,
                depth, num_heads, qkv_bias, ff_expand, do_rate, attn_do_rate,
                drop_path_rate, add_positional_emb, positional_emb_type, cross_rolls):

        super(ST_Transformer, self).__init__()

        self.num_joint = num_joint
        self.num_frame = num_frame
        self.num_p = num_p
        self.add_positional_emb = add_positional_emb
        
        self.dropout = nn.Dropout(p=do_rate)
        self.norm = nn.LayerNorm(dim_joint_emb*num_joint)

        self.emb = nn.Linear(input_channel, dim_joint_emb)
        self.emb_global = nn.Linear(input_channel, dim_joint_emb)
        self.pred_token_emb = nn.Parameter(torch.zeros(1, 1, num_joint, dim_joint_emb))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoder(num_joint, num_frame, num_p, dim_joint_emb, 
            num_heads, ff_expand, qkv_bias, attn_do_rate, do_rate, dpr[i], positional_emb_type, cross_rolls) 
            for i in range(depth)]
        )
        
        self.mlp = nn.Sequential(
                                nn.Linear(dim_joint_emb*num_joint, dim_joint_emb*num_joint),
                                nn.GELU(),
                                nn.LayerNorm(dim_joint_emb*num_joint),
                                nn.Linear(dim_joint_emb*num_joint, dim_joint_emb*num_joint),
                                nn.GELU(),
                                nn.LayerNorm(dim_joint_emb*num_joint),
                                )

    def encoder(self, x, mask=None, tmp_mask=None):

        b, c, f, j = x.shape

        ## generate input embeddings
        x = rearrange(x, 'b c f (j p) -> b c f j p', p=self.num_p)
        x_joints = x[:,:,:,1:,:]
        x_global = x[:,:,:,0,:].unsqueeze(3)

        x_joints = rearrange(x_joints, 'b c f j p -> b f j p c')
        x_global = rearrange(x_global, 'b c f j p -> b f j p c')
        x_joints = self.emb(x_joints)  # joint embedding layer
        x_global = self.emb_global(x_global)  # global translation embedding layer

        x = torch.cat((x_global, x_joints), axis=2)
        x = rearrange(x, 'b f j p c-> b f (j p) c',)
        
        x = self.dropout(x)

        ## GL-Transformer blocks
        for i, block in enumerate(self.encoder_blocks):
            if self.add_positional_emb:
                positional_emb=True
            else:
                positional_emb = False
            if i == 0:
                x_avg = None
            x, x_avg = block(x, mask, positional_emb, tmp_mask, x_avg)

        x = rearrange(x, 'b f j k -> b f (j k)',j=j)
        x = self.norm(x)

        return x


    def forward(self, x):
        
        ## make attention mask for [PAD] tokens
        x_m = x[:,0,:,0]

        tmp_mask = x_m !=999.9

        mask = (x_m != 999.9).unsqueeze(1).repeat(1, x_m.size(1), 1).unsqueeze(1)

        x = self.encoder(x, mask,tmp_mask)

        ## MLP
        x = self.mlp(x) 

        return x
        
