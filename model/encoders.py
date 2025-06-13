import torch
import torch.nn as nn
from utils.config import Configuration


class DynTanhNorm(nn.Module):
    """A toy Dynamic Tanh 'norm' that learns gamma,beta from input features."""

    def __init__(self, embed_dim):
        super(DynTanhNorm, self).__init__()
        self.param_gen = nn.Linear(embed_dim, 2 * embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        # generate gamma, beta: (batch, seq_len, embed_dim) each
        gb = self.param_gen(x)
        gamma, beta = gb.chunk(2, dim=-1)
        # apply tanh-based "normalization"
        out = torch.tanh(gamma * x + beta)
        return out


class StateEncoder(nn.Module):
    """
    - pos_dim: The embedding dimension for positional information (e.g., 256)
    - feat_dim: The dimension of the external features (e.g., 3 for heading, speed, acc)

    Overall idea:
      1. Use a single linear layer: (B, feat_dim) -> (B, 2*pos_dim)
         This projects the external features into 2*pos_dim to generate gamma and beta.
      2. Apply dynamic modulation to pos_emb: tanh( gamma * pos_emb + beta )
         Both gamma and beta have the shape (B, pos_dim).
      3. In this way, the few external features act as control signals to flexibly scale and
       shift the positional information.

    # Note: if we require better non-linear, add L-RELU-L later
    """

    def __init__(self, pos_dim=256, feat_dim=3):
        super(StateEncoder, self).__init__()
        # Generate (gamma, beta) where the input has feat_dim and the output is 2 * pos_dim
        self.param_gen = nn.Linear(feat_dim, 2 * pos_dim)

    def forward(self, pos_emb, features):
        """
        Parameters:
          pos_emb: (B, pos_dim)   -- Positional embedding (already projected to high dimensions)
          features: (B, feat_dim) -- Dynamic features (e.g., heading, speed, acc)

        Output:
          (B, pos_dim) -- The position representation after dynamic modulation
        """
        # Generate gamma and beta using the external features
        gamma_beta = self.param_gen(features)  # shape: (B, 2*pos_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each of shape (B, pos_dim)

        # Apply dynamic activation using tanh
        out = torch.tanh(gamma * pos_emb + beta)  # (B, pos_dim)
        return out


class BackgroundEncoder(nn.Module):
    """
      1) add features into positional information
      2) encode agent topology by BERT ( add context into embedding)
      3) output it as  K, V
    """

    def __init__(self, cfg: Configuration, pos_embed_dim=256, pad_token=0, feat_dim=7, abs_dis_local=5,
                 dropout=0.5, num_layers=1):
        super(BackgroundEncoder, self).__init__()
        # Feature = (heading, v, acc, length, width, abs_dis, hit_dis)
        # heading, v, acc -> speed
        # length, width -> Box size
        # abs_dis -> abstract distance to self
        # abs_dis -> abstract distance to self(hit box level)
        # Question: would it bring noise?
        self.agent_state_encoder = StateEncoder(pos_embed_dim, feat_dim=feat_dim)
        self.pad_token = pad_token
        self.abs_dis_local = abs_dis_local
        self.cfg = cfg

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pos_embed_dim,
            nhead=4,
            dim_feedforward=pos_embed_dim,
            dropout=dropout,
            activation="relu",
            norm_first=True,
        )

        self.topology_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.conv1d_reduce = nn.Conv1d(in_channels=pos_embed_dim,
                                       out_channels=64,
                                       kernel_size=1)
        self.conv1d_1x1 = nn.Conv1d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=1)
        self.conv1d_lowcost = nn.Conv1d(in_channels=64,
                                        out_channels=64,
                                        kernel_size=3,
                                        padding=1,
                                        groups=64)
        self.dim_recovery = nn.Linear(64, pos_embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.cfg.embedding_dim,
            nhead=self.cfg.tf_de_heads,
            dim_feedforward=self.cfg.tf_de_dim,
            dropout=self.cfg.dropout,
            activation='relu'
        )
        self.trajectory_gen = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.output_norm = nn.LayerNorm(self.cfg.embedding_dim)

    def forward(self, agent_emb, agent_feature, goal_emb, agent_mask=None):
        """
        - agent_emb: (batch, time, agent_num, pos_dim)
        - features:  (batch, time, agent_num, feat_dim)
        - goal_emb: (batch, pos_dim)
        - agent_mask: (batch, time, agent_num), 0 => not valid，
          out: (batch, time, agent_num, bert_hidden)
        """
        if self.cfg.simple_deduction:
            bz, t, sl, d_pos = agent_emb.shape
            # ============ (1) Flatten (bz*t*sl, d_pos) ============
            agent_emb_flat = agent_emb.reshape(bz * t * sl, d_pos)  # (batch_size', d_pos)
            agent_feature_flat = agent_feature.reshape(bz * t * sl, -1)  # (batch_size', feat_dim)

            # ============ (2) DyT ============
            #  -> (bz*t*sl, d_pos)
            shifted_agent_emb = self.agent_state_encoder(agent_emb_flat, agent_feature_flat)
            shifted_agent_emb = shifted_agent_emb.reshape(bz, t, sl, -1)
        else:
            bz, sl, d_pos = agent_emb.shape
            # ============ (1) Flatten (bz*sl, d_pos) ============
            agent_emb_flat = agent_emb.reshape(bz * sl, d_pos)  # (batch_size', d_pos)
            agent_feature_flat = agent_feature.reshape(bz * sl, -1)  # (batch_size', feat_dim)

            # ============ (2) DyT ============
            #  -> (bz*sl, d_pos)
            shifted_agent_emb = self.agent_state_encoder(agent_emb_flat, agent_feature_flat)
            shifted_agent_emb = shifted_agent_emb.reshape(bz, sl, -1)

            # ============ implicit_fft ============
            # x = shifted_agent_emb.permute(0, 2, 1)
            # x1 = torch.tanh(self.conv1d_reduce(x))
            # x2 = torch.tanh(self.conv1d_1x1(x1))
            # x = torch.tanh(self.conv1d_lowcost(x1 + x2))
            # x = torch.tanh(self.dim_recovery(x))
            # shifted = x.permute(0, 2, 1)
            #
            # batch = sl * bz
            # #  -> (t, bz*sl, d_pos)
            # T = self.cfg.max_frame + 1
            # mem = shifted.reshape(batch, d_pos).unsqueeze(0)
            #
            # results = [mem[0]]
            # full_mask = torch.triu(torch.ones(T, T, device=shifted.device, dtype=torch.bool), 1)
            # for step in range(1, T):
            #     mask_t = full_mask[:step, :step]
            #     tgt = torch.stack(results, dim=0)  # (t, batch, d_pos)
            #     out = self.trajectory_gen(tgt, mem, tgt_mask=mask_t)
            #     out = self.output_norm(out)
            #     results.append(out[-1])
            # generated = torch.stack(results, dim=0)
            #
            # shifted_agent_emb = generated.reshape(T, bz, sl, d_pos).permute(1, 0, 2, 3)

            # ============ (2) “implicit FFT” 部分改为显式 iFFT ============

            # 2.1 先把 real tensor 转为 complex（假设你的频域信息是实数序列）
            #     如果你已经有复数输入，这一步可以跳过，不要再 to(complex) 一次
            shifted_complex = shifted_agent_emb.to(torch.complex64)  # (bz, sl, d_pos) -> complex

            # 2.2 在 dim=1（长度为 sl 的频率维度）上做一维 iFFT
            #     ifft 返回形状不变，但 dtype 为 complex64
            time_domain_complex = torch.fft.ifft(shifted_complex, n=sl, dim=1)  # (bz, sl, d_pos)

            # 2.3 取实部作为时域特征，丢弃虚部
            time_domain = time_domain_complex.real  # (bz, sl, d_pos), dtype float32

            # 2.4 如果你想做额外的非线性或正则化，可在这里插入，
            #     比如再加一次 tanh 或者 LayerNorm。示例不加，直接进入下一步

            # ============ (3) 自回归生成 (length = max_frame + 1) ============
            # 将时域特征按 batch 展平，送入 TransformerDecoder 生成 T 步
            # 先 reshape 到 (batch, d_pos)
            batch = bz * sl
            mem = time_domain.reshape(batch, d_pos).unsqueeze(0)  # -> (1, batch, d_pos)

            T = self.cfg.max_frame + 1
            results = [mem[0]]
            full_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=shifted_agent_emb.device), diagonal=1)

            # Autoregressive 循环
            for step in range(1, T):
                # 3.1 把已经生成的前 step 步拼成 tgt: (step, batch, d_pos)
                tgt = torch.stack(results, dim=0)

                # 3.2 拿对应尺寸的因果 mask: (step, step)
                tgt_mask = full_mask[:step, :step]

                # 3.3 TransformerDecoder 得到 (step, batch, d_pos)
                out = self.trajectory_gen(tgt, mem, tgt_mask=tgt_mask)  # out: complex? No, decoder 输出是 real
                out = self.output_norm(out)  # 归一化

                # 3.4 拿最后一个 time step 隐状态；若要做 projection 可用 self.proj
                last_hidden = out[-1]  # shape=(batch, d_pos)
                # 如果要投影回 embedding space，可以：
                # next_hidden = self.proj(last_hidden)
                next_hidden = last_hidden

                # 3.5 把 next_hidden 放进 results，作为第 step 步的输出
                results.append(next_hidden)

            # 把 results 中的 T 个 (batch, d_pos) 堆起来： (T, batch, d_pos)
            generated = torch.stack(results, dim=0)

            # ============ (4) reshape 回 (bz, T, sl, d_pos) ============
            # generated: (T, batch, d_pos) 其中 batch = bz*sl
            shifted_agent_emb = generated.view(T, bz, sl, d_pos)  # -> (T, bz, sl, d_pos)
            shifted_agent_emb = shifted_agent_emb.permute(1, 0, 2, 3)

        t = agent_emb.shape[1] if len(agent_emb.shape) == 4 else self.cfg.max_frame + 1
        # ============ (3) goal as a extra agent ============
        goal = goal_emb.unsqueeze(1).repeat(1, t, 1)
        goal = goal.unsqueeze(2)
        # (bz, t, 1, d_pos)

        # ============ (4) add Mask ============
        if agent_mask is not None:
            goal_mask = torch.ones(bz, t, 1, dtype=torch.bool, device=agent_mask.device)
            agent_mask_cat = torch.cat([agent_mask, goal_mask], dim=2)
            agent_mask_flat = agent_mask_cat.reshape(bz * t, sl + 1)
        else:
            agent_mask_flat = None
        topology = torch.cat([shifted_agent_emb, goal], dim=2).reshape(bz * t, sl + 1, d_pos)
        # (bz * t, agent +1, pos_emb)
        topology_emb = self.topology_encoder(topology.transpose(0, 1), src_key_padding_mask=~agent_mask_flat)
        topology_emb = topology_emb.transpose(0, 1)
        topology_emb = topology_emb.reshape(bz, t, sl + 1, d_pos)
        return topology_emb
