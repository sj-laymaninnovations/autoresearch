import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig

def weight_quant(weight):
    # BitNet 1.58b weight quantization
    scale = weight.abs().mean().clamp(min=1e-5)
    weight_scaled = weight / scale
    weight_q = torch.clamp(torch.round(weight_scaled), -1, 1)
    return weight_q.detach() - weight.detach() + weight, scale

def activation_quant(x):
    # BitNet 1.58b activation quantization (8-bit)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_scaled = (x / scale) * 127.0
    x_q = torch.clamp(torch.round(x_scaled), -128, 127)
    return x_q.detach() - x.detach() + x, scale

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.ln = nn.LayerNorm(in_features)
        
    def forward(self, x):
        x = self.ln(x)
        x_q, scale_x = activation_quant(x)
        w_q, scale_w = weight_quant(self.weight)
        y = F.linear(x_q, w_q, self.bias)
        return y * (scale_w * scale_x / 127.0)

# The rest of the architecture uses BitLinear instead of nn.Linear
class BitMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc   = BitLinear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.c_proj = BitLinear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))

class BitCausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head  = cfg.n_head
        self.d_head  = cfg.d_model // cfg.n_head
        self.d_model = cfg.d_model

        self.c_attn  = BitLinear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.c_proj  = BitLinear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        
        # simplified RoPE for demonstration
        self.register_buffer("bias", torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).view(1, 1, cfg.seq_len, cfg.seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) / scale
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))

class BitBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = BitCausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.mlp  = BitMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MTPDrafter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.ln = nn.LayerNorm(cfg.d_model)
        
    def forward(self, h_t, next_token_emb):
        x = torch.cat([h_t, next_token_emb], dim=-1)
        x = self.proj(x)
        x = F.gelu(x)
        return self.ln(x)

class BitGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(cfg.vocab_size, cfg.d_model),
            "h": nn.ModuleList([BitBlock(cfg) for _ in range(cfg.n_layer)]),
            "ln_f": nn.LayerNorm(cfg.d_model),
        })
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        # MTP configuration
        self.mtp_depth = getattr(cfg, "mtp_depth", 1)
        if self.mtp_depth > 1:
            self.drafters = nn.ModuleList([
                MTPDrafter(cfg) for _ in range(self.mtp_depth - 1)
            ])

    def forward(self, idx, targets=None):
        b, t = idx.size()
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        h_base = self.transformer.ln_f(x)
        
        logits_main = self.lm_head(h_base)
        
        if targets is None or self.mtp_depth <= 1:
            if targets is not None:
                loss = F.cross_entropy(logits_main.view(-1, logits_main.size(-1)), targets.view(-1), ignore_index=-1)
                return logits_main, loss
            return logits_main[:, [-1], :], None
            
        # MTP Training
        total_loss = F.cross_entropy(logits_main.view(-1, logits_main.size(-1)), targets.view(-1), ignore_index=-1)
        h_current = h_base[:, :-1, :]
        
        for i, drafter in enumerate(self.drafters):
            next_actual_token_idx = idx[:, 1+i:]
            if next_actual_token_idx.size(1) == 0:
                break
                
            next_emb = self.transformer.wte(next_actual_token_idx)
            h_current = h_current[:, :next_emb.size(1), :] 
            h_next = drafter(h_current, next_emb)
            
            drafter_logits = self.lm_head(h_next)
            drafter_targets = targets[:, 1+i:]
            
            loss_k = F.cross_entropy(drafter_logits.reshape(-1, drafter_logits.size(-1)), drafter_targets.reshape(-1), ignore_index=-1)
            total_loss += (loss_k * 0.5) # Downweight auxiliary loss
            
            h_current = h_next
            
        return logits_main, total_loss / self.mtp_depth

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

