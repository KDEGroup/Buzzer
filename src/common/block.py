import math
import torch.nn as nn  
import torch 


class Attn(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, n_head=12) -> None:
        super(Attn, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_head = n_head
        
        self.Wq = nn.Linear(self.input_size, self.n_head * self.hidden_size)
        self.Wk = nn.Linear(self.input_size, self.n_head * self.hidden_size)
        self.Wv = nn.Linear(self.input_size, self.n_head * self.hidden_size)
        
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.linear_merge = nn.Linear(self.n_head * self.hidden_size, self.hidden_size)
        
    def forward(self, input_embed, attention_mask):
        bs = input_embed.size(0)
        q = self.Wq(input_embed).view(bs, -1, self.n_head, self.hidden_size).transpose(1, 2)
        k = self.Wv(input_embed).view(bs, -1, self.n_head, self.hidden_size).transpose(1, 2)
        v = self.Wv(input_embed).view(bs, -1, self.n_head, self.hidden_size).transpose(1, 2)
        
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hidden_size)
        # if attention_mask is not None:
        #     # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        score.masked_fill_(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        attn = nn.Softmax(dim=-1)(score)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_head * self.hidden_size)
        context = self.linear_merge(context)
        context = self.layer_norm(context)
        
        # [bs, seqlen, hidden_size, ]  [bs, seqlen]
        context.masked_fill_(attention_mask.unsqueeze(2) == 0, 0)
        context_s = context.sum(dim = 1) / attention_mask.sum(dim=1).unsqueeze(1)
        # breakpoint()
        return context_s 


def minmax_norm_2d_feature(feature, attention_mask,  min_=None, max_=None, type_='seq'):
    '''
    计算attention_mask不为0位置的均值和方差，并归一化。
    attention_mask为0位置的不做考虑
    '''
    if min_ is None and max_ is None:
        if type_ == 'seq':
            attend_element = torch.masked_select(feature, attention_mask)
        else:
            attend_element = feature
            
        max_ = torch.max(attend_element)
        min_ = torch.min(attend_element)

        feature = (feature - min_) / (max_ - min_) 
        if type_ == 'seq':
            feature = feature.masked_fill(~attention_mask, 0)
        
        return feature, min_, max_ 
    
    else:
        feature = (feature - min_) / (max_ - min_) 
        if type_ == 'seq':
            feature = feature.masked_fill(~attention_mask, 0)
        
        return feature