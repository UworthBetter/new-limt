# LiMT 模型：MRE(式1-14) + LDL(式15-20)
# 输入 x: [B, S, T, F]（B 个交易日一批），pad_mask: [B,S] bool（True=填充位）
# 注意力用 F.scaled_dot_product_attention 实现（与 nn.MultiheadAttention 数学等价，
# 融合内核前向/反向都快得多）
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPA(nn.Module):
    """多头注意力：x [B,N,d]（可带 key mask [N]，True=填充）→ [B,N,d]"""

    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, key_invalid=None):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]          # 各 [B,H,N,dk]
        am = None
        if key_invalid is not None:
            if key_invalid.dim() == 1:
                am = (~key_invalid).view(1, 1, 1, N)      # [1,1,1,N] 广播
            else:
                am = (~key_invalid).unsqueeze(1).unsqueeze(1)  # [B,1,1,N] 广播
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am)   # 式3-4/9-10
        o = o.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.out(o)


class MRE(nn.Module):
    """市场状态编码器：先跨股票注意力(每个时间步)，后股票内时间注意力"""

    def __init__(self, d_in, d=256, head_cross=4, head_temp=2, dropout=0.0,
                 no_cross=False):
        super().__init__()
        self.no_cross = no_cross            # 消融：w/o Cross-Stock Modeling
        self.proj = nn.Linear(d_in, d)                                   # 式1
        self.cross_attn = SDPA(d, head_cross)                            # 式3-5
        self.ln_cross = nn.LayerNorm(d)                                  # 式6
        self.temp_attn = SDPA(d, head_temp)                              # 式9-11
        self.ln_temp = nn.LayerNorm(d)                                   # 式12

    def forward(self, x, pad_mask):
        # x: [B,S,T,F] → [B,S,d]
        B, S, T, _ = x.shape
        h = self.proj(x)                                                # 式1
        if not self.no_cross:
            # --- 跨股票注意力：每个 (day,time) 在 S 维做注意力 ---
            hc = h.permute(0, 2, 1, 3).reshape(B * T, S, -1)             # [B*T,S,d]
            kinv = pad_mask.unsqueeze(1).expand(B, T, S).reshape(-1, S)  # [B*T,S]
            attn_out = self.cross_attn(hc, kinv)                         # 式3-5
            h = self.ln_cross(hc + attn_out)                             # 式6
            h = h.reshape(B, T, S, -1).permute(0, 2, 1, 3)               # [B,S,T,d] 式7
        # --- 时间注意力：每只股票在 T 维做注意力（batch=B*S）---
        ht = h.reshape(B * S, T, -1)                                     # [B*S,T,d]
        attn_out = self.temp_attn(ht)                                    # 式9-11
        h = self.ln_temp(ht + attn_out)                                  # 式12
        h = h.reshape(B, S, T, -1)
        return h[:, :, -1, :]                                            # 式13: 最后时间步


class LDL(nn.Module):
    """流动性驱动学习：MMoE(式15-17) + 跨任务知识转移(式18-19) + 预测头(式20)"""

    def __init__(self, d=256, n_experts=4, no_ldl=False):
        super().__init__()
        self.no_ldl = no_ldl              # 消融：w/o LDL（去掉跨任务知识转移项）
        self.n = n_experts
        self.experts = nn.ModuleList([nn.Linear(d, d) for _ in range(n_experts)])  # 式16
        self.gates = nn.ModuleDict({k: nn.Linear(d, n_experts) for k in "rvs"})     # 式15
        self.towers = nn.ModuleDict({k: nn.Linear(d, d // 2) for k in "rvs"})       # 式17
        self.beta_v = nn.Linear(d // 2, 1)             # 式18
        self.beta_s = nn.Linear(d // 2, 1)
        self.head_r = nn.Linear(d // 2, 1)             # 式20
        self.head_v = nn.Linear(d // 2, 1)
        self.head_s = nn.Linear(d // 2, 1)

    def forward(self, h):
        # h: [B,S,d] → 输出各 [B,S]
        exp_out = torch.stack([e(h) for e in self.experts], dim=-2)    # [B,S,n,d]
        outs, G = {}, {}
        for k in "rvs":
            alpha = torch.softmax(self.gates[k](h), dim=-1)            # [B,S,n] 式15
            outs[k] = (exp_out * alpha.unsqueeze(-1)).sum(-2)          # [B,S,d] 式16
            G[k] = torch.relu(self.towers[k](outs[k]))                 # [B,S,d/2] 式17
        bv = torch.sigmoid(self.beta_v(G["v"]))                        # [B,S,1] 式18
        bs = torch.sigmoid(self.beta_s(G["s"]))
        if self.no_ldl:                                                # 消融：w/o LDL
            o_r = G["r"]
        else:
            o_r = G["r"] + bv * G["v"] + bs * G["s"]                   # [B,S,d/2] 式19
        return (self.head_r(o_r).squeeze(-1),                          # ẑ_r
                self.head_v(G["v"]).squeeze(-1),
                self.head_s(G["s"]).squeeze(-1))


class LiMT(nn.Module):
    def __init__(self, d_in, d=256, head_cross=4, head_temp=2, n_experts=4,
                 dropout=0.0, no_cross=False, no_ldl=False):
        super().__init__()
        self.mre = MRE(d_in, d, head_cross, head_temp, dropout, no_cross)
        self.ldl = LDL(d, n_experts, no_ldl)

    def forward(self, x, pad_mask):
        h = self.mre(x, pad_mask)
        return self.ldl(h)      # (ẑ_r, ẑ_v, ẑ_σ) 各 [B,S]
