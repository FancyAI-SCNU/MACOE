import numpy as np
import torch
from torch import nn
from itertools import chain
import torch.nn.functional as F
import math
EPS = 1e-8

np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        
        super(RMSNorm, self).__init__()
        
        self.eps = eps

    def forward(self, x,gamma):
        
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  
        x_norm = x / rms  
        return gamma * x_norm  
    
class MLA(nn.Module):
    def __init__(
        self,
        device
    ) :
        super(MLA, self).__init__()
        self.n_agents = 8
        self.agent_stocks = 10
        self.d_model = 128
        self.p_dim = 64
        self.obs_dim = 32
        self.act_dim = 5
        self.msg_dims = 80
        self.kv_lora_rank = 512
        self.device= device
        self.n_heads = self.n_agents

                         
        self.pool= nn.AdaptiveAvgPool1d(1)  

        self.wq_b = nn.Linear((self.obs_dim+self.act_dim)*self.n_agents,(self.d_model*self.n_heads+self.p_dim)*self.n_agents, bias=False)
        
        self.wkv_a = nn.Linear(self.msg_dims, self.kv_lora_rank+self.p_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.d_model + self.d_model)*self.n_agents)
         
        self.msg_pro = nn.Linear(self.n_heads * self.d_model, self.msg_dims)
         
        self.gamma_ = nn.Parameter(torch.ones(self.kv_lora_rank)).to(self.device)

        self.freq = self.precompute_freqs_cis().to(self.device)
        self.softmax_scale =(( self.obs_dim+self.act_dim)+self.d_model)**0.5 


    
    def precompute_freqs_cis(self):
        
        dim = 64
        seqlen = 1
        beta_fast = 32
        beta_slow = 1
        base = 10000.0
        factor = 40

        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            
            return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim-1)

        def linear_ramp_factor(min, max, dim):
            if min == max:
                max += 0.001
            linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if seqlen > 4:
            low, high = find_correction_range(beta_fast, beta_slow, dim, base, 4096)
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        t = torch.arange(seqlen)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def apply_rotary_emb(self,x, freqs_cis):
        dtype = x.dtype
        x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
        y = torch.view_as_real(x * freqs_cis).flatten(3)
        return y.to(dtype)


    def forward(self,f_0_input_all_agents_new,pi_all_agents_new,msg_prev):
        if len(pi_all_agents_new.shape)!=3:
            pi_all_agents_new = pi_all_agents_new.unsqueeze(1)
        if len(f_0_input_all_agents_new.shape)!=3:
            f_0_input_all_agents_new = f_0_input_all_agents_new.squeeze(2)
            batch_num = f_0_input_all_agents_new.shape[0]
        else:
            batch_num=1
        
         
        q_obs_a = torch.concat([f_0_input_all_agents_new,pi_all_agents_new],dim=-1)
        q_obs_a = q_obs_a.reshape(batch_num,-1,self.n_agents*q_obs_a.shape[-1])
         
        if len(msg_prev.shape)==1:
            kv = msg_prev.unsqueeze(0).unsqueeze(0)
        else:
            kv = msg_prev.unsqueeze(1)
         
        q = self.wq_b(q_obs_a)
         
        q,q_pos = torch.split(q,[(self.d_model*self.n_heads)*self.n_agents,self.p_dim*self.n_agents],dim=-1)
        q = q.reshape(batch_num,self.n_agents,self.n_heads,-1)
        q_pos = q_pos.reshape(batch_num,self.n_agents,self.n_heads,-1)
        
        q_pos = self.apply_rotary_emb(q_pos.unsqueeze(2), self.freq)
        
         
        kv = self.wkv_a(kv)
        kv,k_pos = torch.split(kv,[self.kv_lora_rank,self.p_dim],dim=-1)
        k_pos = self.apply_rotary_emb(k_pos.unsqueeze(2), self.freq)
        kv = self.wkv_b(self.kv_norm(kv,self.gamma_))
         
        k,v = torch.split(kv,self.n_heads*(self.d_model)*self.n_agents,dim=-1)
         
        k = k.reshape(q.shape[0],q.shape[1],self.n_heads,-1)
        v = v.reshape(q.shape[0],q.shape[1],self.n_heads,-1)
        q = q.reshape(q.shape[0],q.shape[1],self.n_heads,-1)
        
        q = torch.cat([q,q_pos],dim=-1)
        
        k = torch.cat([k, k_pos.expand(-1, self.n_agents, self.n_heads, -1)], dim=-1)
         
        
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  

        scale = self.softmax_scale
        attn_scores = attn_scores / scale

        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        head_outputs = output.sum(dim=2)
        output = head_outputs.reshape(batch_num, -1)
      
        msg = output.reshape(batch_num,-1)
        
        msg = self.msg_pro(msg)
         
        if batch_num>1:
            msg = msg.squeeze(1)
        else:
            msg = msg.squeeze(0).squeeze(0)
         
        return msg
