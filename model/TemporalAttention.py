import torch
import torch.nn as nn
import torch.nn.functional as F_func
from .net import Unit2D
import math
import numpy as np
import time

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dropout = False
scale_norm = False
save = False
multi_matmul = False

''' Class that implements Temporal Transformer.
Function adapted from: https://github.com/leaderj1001/Attention-Augmented-Conv2d
'''


class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        # pe = position/position.max()*2 -1
        # pe = pe.view(time_len, joint_num).unsqueeze(0).unsqueeze(0)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x


class TemporalAttention(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size_temporal=9):
        super(TemporalAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.visualization = False
        self.num_point = 25
        self.more_channels = False
        self.only_temporal_att = False
        self.drop_connect = True
        self.kernel_size_temporal = 9
        self.num = 4
        self.more_relative = False
        dv_factor = 0.25
        self.dk = int(dv_factor * out_channels)
        self.Nh = 8
        self.bn_flag = True
        self.shape = 25
        self.relative = False
        self.stride = stride
        self.padding = (self.kernel_size_temporal - 1) // 2
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.data_normalization = True
        self.skip_conn = True

        if not self.only_temporal_att:
            self.dv = int(dv_factor * out_channels)
        else:
            self.dv = out_channels
        if (self.in_channels != self.out_channels) or (stride != 1):
            self.down = Unit2D(
                self.in_channels, self.out_channels, kernel_size=1, stride=stride)
        else:
            self.down = None
        if self.data_normalization:
            self.data_bn = nn.BatchNorm1d(self.in_channels * self.num_point)
        if dropout:
            self.dropout = nn.Dropout(0.25)

        # Temporal convolution
        if (not self.only_temporal_att):
            self.tcn_conv = Unit2D(in_channels, out_channels - self.dv, dropout=dropout,
                                   kernel_size=kernel_size_temporal,
                                   stride=self.stride)
        if (self.more_channels):

            self.qkv_conv = nn.Conv2d(self.in_channels, (3 * self.dk + self.dv) * self.Nh // self.num,
                                      kernel_size=(1, stride),
                                      stride=(1, stride),
                                      padding=(0, int((1 - 1) / 2)))
        else:
            if self.num_point % 2 != 0:
                self.qkv_conv = nn.Conv2d(self.in_channels, 3 * self.dk + self.dv, kernel_size=(1, stride),
                                          stride=(1, stride),
                                          padding=(0, int((1 - 1) / 2)))
            else:
                self.qkv_conv = nn.Conv2d(self.in_channels, 3 * self.dk + self.dv, kernel_size=(1, 1),
                                          stride=(1, stride),
                                          padding=(0, int((1 - 1) / 2)))
        if (self.more_channels):
            self.attn_out = nn.Conv2d(self.dv * self.Nh // self.num, self.dv, kernel_size=1, stride=1)
        else:
            self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.out_channels == 64:
            self.frames = 64

        if self.out_channels == 128:
            self.frames = 32

        if self.out_channels == 256:
            self.frames = 16

        if self.relative:
            if self.more_channels:
                self.key_rel = nn.Parameter(
                    torch.randn((2 * self.frames - 1, self.dk // self.num), requires_grad=True))

            else:
                self.key_rel = nn.Parameter(
                    torch.randn((2 * self.frames - 1, self.dk // self.Nh), requires_grad=True))

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"

        rr = 2
        self.fc1c = nn.Linear(self.out_channels, self.out_channels // rr)
        self.fc2c = nn.Linear(self.out_channels // rr, self.out_channels)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self.pes = PositionalEncoding(in_channels, self.shape, self.frames, 'temporal')

    def forward(self, x, STTR=False):

        if STTR:
            # ===== self-attention (ST-TR) ======
            # Input x ==> (batch_size, channels, time, joints)
            N, C, T, V = x.size()
            x_sum = x
            if self.data_normalization:
                x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
                x = self.data_bn(x)
                x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)

                # Joint dimension is put inside the batch, in order to process each joint along the time separately
            x = x.permute(0, 3, 1, 2).reshape(-1, C, 1, T)
            flat_q, flat_k, flat_g, flat_v, q, k, g,  v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
            B, self.Nh, C, T = flat_q.size()
            logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
            weights = self.tan(logits)
            # attn_out ==> (batch, Nh, time, dvh)
            # weights*V ==>(batch, Nh, time, time)*(batch, Nh, time, dvh)=(batch, Nh, time, dvh)
            attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
            attn_out = torch.reshape(attn_out, (B, self.Nh, 1, T, -1))
            attn_out = attn_out.permute(0, 1, 4, 2, 3)

            # combine_heads_2d, combine heads only after having calculated each Z separately
            # (batch, Nh*dv, time, 1)
            attn_out = self.combine_heads_2d(attn_out)

            # Multiply for W0 (batch, out_channels, time, joints) with out_channels=dv
            attn_out = self.attn_out(attn_out)
            attn_out = attn_out.reshape(N, V, -1, T).permute(0, 2, 3, 1)

            if self.skip_conn:
                if dropout:
                    attn_out = self.dropout(attn_out)

                    if not self.only_temporal_att:
                        x = self.tcn_conv(x_sum)
                        result = torch.cat((x, attn_out), dim=1)
                    else:
                        result = attn_out

                    result = result + (x_sum if (self.down is None) else self.down(x_sum))
                else:
                    if not self.only_temporal_att:
                        x = self.tcn_conv(x_sum)
                        result = torch.cat((x, attn_out), dim=1)
                    else:
                        result = attn_out

                    result = result + (x_sum if (self.down is None) else self.down(x_sum))
            else:
                result = attn_out

            if self.bn_flag:
                result = self.bn(result)
            result = self.relu(result)
            return result

        else:
            # ==== PAT-Net ==============
            # Input x
            # (batch_size, channels, time, joints)
            N, C, T, V = x.size()
            y = self.pes(x)
            x_sum = x

            if self.data_normalization:
                x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
                x = self.data_bn(x)
                x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)

            # Joint dimension is put inside the batch, in order to process each joint along the time separately
            x = x.permute(0, 3, 1, 2).reshape(-1, C, 1, T)
            y = y.permute(0, 3, 1, 2).reshape(-1, C, 1, T)
            if scale_norm:
                self.scale = ScaleNorm(scale=C ** 0.5)
                x = self.scale(x)

            flat_q, flat_k,  flat_g, flat_v, q, k, g,  v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
            pos_qp, pos_kp, po_gp, pos_vp, qp, kp, gp, vp = self.position_projection(y, self.dk, self.dv, self.Nh)

            B, self.Nh, C, T = flat_q.size()
            mu_Q = flat_q.mean(-1).unsqueeze(-1)  # [1, D']
            mu_K = flat_k.mean(-1).unsqueeze(-1)  # [1, D']
            flat_q = flat_q - mu_Q
            flat_kq = flat_k - mu_K

            # Calculate the scores, obtained by doing q*k
            # (batch_size, Nh, time, dkh)*(batch_size, Nh,dkh, time) =  (batch_size, Nh, time, time)
            logits_position = torch.matmul(pos_qp.transpose(2, 3), pos_kp)
            pairwise = torch.matmul(flat_q.transpose(2, 3), flat_k)
            unary = torch.matmul(mu_Q.transpose(2, 3), flat_k)

            # ==========Relative positional encoding is used or not  ===============
            if self.relative:
                rel_logits = self.relative_logits(q)
                logits_sum_pair = torch.add(pairwise, rel_logits)
                logits_sum_unary = torch.add(unary, rel_logits)
                weights_pair_sum = self.tan(logits_sum_pair + logits_position)
                weights_unary_sum = self.tan(logits_sum_unary + logits_position)
                weights = torch.add(weights_pair_sum, weights_unary_sum)
            # ========================================================================#
            else:
                weights_pair = self.tan(pairwise + logits_position)
                weights_unary = self.tan(unary + logits_position)
                weights = torch.add(weights_pair, weights_unary)

            attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
            attn_out = torch.reshape(attn_out, (B, self.Nh, 1, T, -1))
            attn_out = attn_out.permute(0, 1, 4, 2, 3)

            # combine_heads_2d, combine heads only after having calculated each Z separately
            # (batch, Nh*dv, time, 1)
            attn_out = self.combine_heads_2d(attn_out)

            # Multiply for W0 (batch, out_channels, time, joints) with out_channels=dv
            attn_out = self.attn_out(attn_out)
            attn_out = attn_out.reshape(N, V, -1, T).permute(0, 2, 3, 1)

            if self.skip_conn:
                if dropout:
                    attn_out = self.dropout(attn_out)

                    if not self.only_temporal_att:
                        x = self.tcn_conv(x_sum)
                        result = torch.cat((x, attn_out), dim=1)
                    else:
                        result = attn_out

                    result = result + (x_sum if (self.down is None) else self.down(x_sum))


                else:
                    if not self.only_temporal_att:
                        x = self.tcn_conv(x_sum)
                        result = torch.cat((x, attn_out), dim=1)
                    else:
                        result = attn_out

                    result = result + (x_sum if (self.down is None) else self.down(x_sum))


            else:
                result = attn_out

            if self.bn_flag:
                result = self.bn(result)
            y = self.relu(result)

            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

            return y

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)

        # In this case V=1, because Temporal Transformer is applied for each joint separately
        N, C, V1, T1 = qkv.size()
        q, k, g, v = torch.split(qkv, [dk, dk, dk, dv], dim=1)

        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        g = self.split_heads_2d(g, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q * (dkh ** -0.5)

        flat_q = torch.reshape(q, (N, Nh, dkh, V1 * T1))
        flat_k = torch.reshape(k, (N, Nh, dkh, V1 * T1))
        flat_g = torch.reshape(g, (N, Nh, dkh, V1 * T1))
        flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, V1 * T1))
        return flat_q, flat_k, flat_g, flat_v, q, k, g, v

    def position_projection(self, pos, dk, dv, Nh):
        pqk = self.qkv_conv(pos)
        N, _, T, V = pqk.size()
        q_pos, k_pos, g_pos, v_pos = torch.split(pqk, [dk, dk, dk, dv], dim=1)
        q = self.split_heads_2d(q_pos, Nh)
        k = self.split_heads_2d(k_pos, Nh)
        g = self.split_heads_2d(g_pos, Nh)
        v = self.split_heads_2d(v_pos, Nh)

        dkh = dk // Nh
        pos_q = q * (dkh ** -0.5)

        pos_q = torch.reshape(q, (N, Nh, dkh, T * V))
        pos_k = torch.reshape(k, (N, Nh, dkh, T * V))
        pos_g = torch.reshape(g, (N, Nh, dkh, T * V))
        pos_v = torch.reshape(v, (N, Nh, dv // self.Nh, T * V))
        return pos_q, pos_k, pos_g, pos_v, q, k, g, v

    def split_heads_2d(self, x, Nh):
        B, channels, F, V = x.size()
        ret_shape = (B, Nh, channels // Nh, F, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, F, V = x.size()
        ret_shape = (batch, Nh * dv, F, V)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, _, T = q.size()
        # B, Nh, V, T, dk -> B, Nh, F, 1, dk
        q = q.permute(0, 1, 3, 4, 2)
        q = q.reshape(B, Nh, T, dk)
        rel_logits = self.relative_logits_1d(q, self.key_rel)
        return rel_logits

    def relative_logits_1d(self, q, rel_k):
        # compute relative logits along one dimension
        # (B, Nh,  1, V, channels // Nh)*(2 * K - 1, self.dk // Nh)
        # (B, Nh,  1, V, 2 * K - 1)
        rel_logits = torch.einsum('bhld,md->bhlm', q, rel_k)
        rel_logits = self.rel_to_abs(rel_logits)
        B, Nh, L, L = rel_logits.size()
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()
        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


class ScaleNorm(nn.Module):
    """ScaleNorm"""

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = scale

        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=1, keepdim=True).clamp(min=self.eps)
        return x * norm
