# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# multi_matmul = False
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, in_channels, out_c, A, padding=0, kernel_size=1, shape=25, stride=1, t_dilation=1):
#         super(SpatialAttention, self).__init__()
#
#         self.in_channels = in_channels
#         self.complete = True
#         self.kernel_size = 1
#         # self.dk = int(out_c * 0.25)
#         # self.dv = int(out_c * 0.25)
#         self.dv = 0.25
#         self.dk = 0.25
#         self.num = 4
#         self.layer = 0
#         self.more_channels = False
#         self.drop_connect = False
#         self.visualization = False
#         self.data_normalization = True
#         self.skip_conn = True
#         self.adjacency = False
#         self.Nh = 8
#         self.num_point = 25
#         self.A = A[0] + A[1] + A[2]
#         if self.adjacency:
#             self.mask = nn.Parameter(torch.ones(self.A.size()))
#         self.shape = shape
#         self.relative = False
#         self.last_graph = False
#         self.stride = stride
#         self.padding = (self.kernel_size - 1) // 2
#
#         self.dk = int(out_c * self.dk)
#         self.dv = int(out_c)
#
#         assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
#         assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
#         assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
#         assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."
#
#         if (self.more_channels):
#
#             self.qkv_conv = nn.Conv2d(self.in_channels, (2 * self.dk + self.dv) * self.Nh // self.num,
#                                       kernel_size=self.kernel_size,
#                                       stride=stride,
#                                       padding=self.padding)
#         else:
#             self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size,
#                                       stride=stride,
#                                       padding=self.padding)
#         if (self.more_channels):
#
#             self.attn_out = nn.Conv2d(self.dv * self.Nh // self.num, self.dv, kernel_size=1, stride=1)
#         else:
#             self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)
#
#         if self.relative:
#             # Two parameters are initialized in order to implement relative positional encoding
#             # One weight repeated over the diagonal
#             # V^2-V+1 paramters in positions outside the diagonal
#             if self.more_channels:
#                 self.key_rel = nn.Parameter(
#                     torch.randn(((self.num_point ** 2) - self.num_point, self.dk // self.num), requires_grad=True))
#             else:
#                 self.key_rel = nn.Parameter(
#                     torch.randn(((self.num_point ** 2) - self.num_point, self.dk // Nh), requires_grad=True))
#             if self.more_channels:
#                 self.key_rel_diagonal = nn.Parameter(torch.randn((1, self.dk // self.num), requires_grad=True))
#             else:
#                 self.key_rel_diagonal = nn.Parameter(torch.randn((1, self.dk // self.Nh), requires_grad=True))
#
#     def forward(self, x):
#         # Input x
#         # (batch_size, channels, 1, joints)
#         B, _, T, V = x.size()
#
#         # flat_q, flat_k, flat_v
#         # (batch_size, Nh, dvh or dkh, joints)
#         # dvh = dv / Nh, dkh = dk / Nh
#         # q, k, v obtained by doing 2D convolution on the input (q=XWq, k=XWk, v=XWv)
#         flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
#
#         # Calculate the scores, obtained by doing q*k
#         # (batch_size, Nh, joints, dkh)*(batch_size, Nh, dkh, joints) =  (batch_size, Nh, joints,joints)
#         # The multiplication can also be divided (multi_matmul) in case of space problems
#         if (multi_matmul):
#             for i in range(0, 5):
#                 flat_q_5 = flat_q[:, :, :, (5 * i):(5 * (i + 1))]
#                 product = torch.matmul(flat_q_5.transpose(2, 3), flat_k)
#                 if (i == 0):
#                     logits = product
#                 else:
#                     logits = torch.cat((logits, product), dim=2)
#         else:
#             logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
#
#         # In this version, the adjacency matrix is weighted and added to the attention logits of transformer to add
#         # information of the original skeleton structure
#         if (self.adjacency):
#             self.A = self.A.cuda(device)
#             logits = logits.reshape(-1, V, V)
#             M, V, V = logits.shape
#             A = self.A
#             A *= self.mask
#             A = A.unsqueeze(0).expand(M, V, V)
#             logits = logits + A
#             logits = logits.reshape(B, self.Nh, V, V)
#
#         # Relative positional encoding is used or not
#         if self.relative:
#             rel_logits = self.relative_logits(q)
#             logits_sum = torch.add(logits, rel_logits)
#
#         # Calculate weights
#         if self.relative:
#             weights = F.softmax(logits_sum, dim=-1)
#         else:
#             weights = F.softmax(logits, dim=-1)
#
#         # Drop connect implementation to avoid overfitting
#         if (self.drop_connect and self.training):
#             mask = torch.bernoulli((0.5) * torch.ones(B * self.Nh * V, device=device))
#             mask = mask.reshape(B, self.Nh, V).unsqueeze(2).expand(B, self.Nh, V, V)
#             weights = weights * mask
#             weights = weights / (weights.sum(3, keepdim=True) + 1e-8)
#
#         # attn_out
#         # (batch, Nh, joints, dvh)
#         # weights*V
#         # (batch, Nh, joints, joints)*(batch, Nh, joints, dvh)=(batch, Nh, joints, dvh)
#         attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
#
#         if not self.more_channels:
#             attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.Nh))
#         else:
#             attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.num))
#
#         attn_out = attn_out.permute(0, 1, 4, 2, 3)
#
#         # combine_heads_2d, combine heads only after having calculated each Z separately
#         # (batch, Nh*dv, 1, joints)
#         attn_out = self.combine_heads_2d(attn_out)
#
#         # Multiply for W0 (batch, out_channels, 1, joints) with out_channels=dv
#         attn_out = self.attn_out(attn_out)
#         return attn_out
#
#     def compute_flat_qkv(self, x, dk, dv, Nh):
#         qkv = self.qkv_conv(x)
#         # T=1 in this case, because we are considering each frame separately
#         N, _, T, V = qkv.size()
#
#         # if self.more_channels=True, to each head is assigned dk*self.Nh//self.num channels
#         if self.more_channels:
#             q, k, v = torch.split(qkv, [dk * self.Nh // self.num, dk * self.Nh // self.num, dv * self.Nh // self.num],
#                                   dim=1)
#         else:
#             q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
#         q = self.split_heads_2d(q, Nh)
#         k = self.split_heads_2d(k, Nh)
#         v = self.split_heads_2d(v, Nh)
#
#         dkh = dk // Nh
#         q = q * (dkh ** -0.5)
#         if self.more_channels:
#             flat_q = torch.reshape(q, (N, Nh, dk // self.num, T * V))
#             flat_k = torch.reshape(k, (N, Nh, dk // self.num, T * V))
#             flat_v = torch.reshape(v, (N, Nh, dv // self.num, T * V))
#         else:
#             flat_q = torch.reshape(q, (N, Nh, dkh, T * V))
#             flat_k = torch.reshape(k, (N, Nh, dkh, T * V))
#             flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, T * V))
#         return flat_q, flat_k, flat_v, q, k, v
#
#     def split_heads_2d(self, x, Nh):
#         B, channels, T, V = x.size()
#         ret_shape = (B, Nh, channels // Nh, T, V)
#         split = torch.reshape(x, ret_shape)
#         return split
#
#     def combine_heads_2d(self, x):
#         batch, Nh, dv, T, V = x.size()
#         ret_shape = (batch, Nh * dv, T, V)
#         return torch.reshape(x, ret_shape)
#
#     def relative_logits(self, q):
#         B, Nh, dk, T, V = q.size()
#         q = torch.transpose(q, 2, 4).transpose(2, 3)
#         q_first = q.unsqueeze(4).expand((B, Nh, T, V, V - 1, dk))
#         q_first = torch.reshape(q_first, (B * Nh * T, -1, dk))
#
#         # q used to multiply for the embedding of the parameter on the diagonal
#         q = torch.reshape(q, (B * Nh * T, V, dk))
#         # key_rel_diagonal: (1, dk) -> (V, dk)
#         param_diagonal = self.key_rel_diagonal.expand((V, dk))
#         rel_logits = self.relative_logits_1d(q_first, q, self.key_rel, param_diagonal, T, V, Nh)
#         return rel_logits
#
#     def relative_logits_1d(self, q_first, q, rel_k, param_diagonal, T, V, Nh):
#         # compute relative logits along one dimension
#         # (B*Nh*1,V^2-V, self.dk // Nh)*(V^2 - V, self.dk // Nh)
#
#         # (B*Nh*1, V^2-V)
#         rel_logits = torch.einsum('bmd,md->bm', q_first, rel_k)
#         # (B*Nh*1, V)
#         rel_logits_diagonal = torch.einsum('bmd,md->bm', q, param_diagonal)
#
#         # reshapes to obtain Srel
#         rel_logits = self.rel_to_abs(rel_logits, rel_logits_diagonal)
#
#         rel_logits = torch.reshape(rel_logits, (-1, Nh, V, V))
#         return rel_logits
#
#     def rel_to_abs(self, rel_logits, rel_logits_diagonal):
#         B, L = rel_logits.size()
#         B, V = rel_logits_diagonal.size()
#
#         # (B, V-1, V) -> (B, V, V)
#         rel_logits = torch.reshape(rel_logits, (B, V - 1, V))
#         row_pad = torch.zeros(B, 1, V).to(rel_logits)
#         rel_logits = torch.cat((rel_logits, row_pad), dim=1)
#
#         # concat the other embedding on the left
#         # (B, V, V) -> (B, V, V+1) -> (B, V+1, V)
#         rel_logits_diagonal = torch.reshape(rel_logits_diagonal, (B, V, 1))
#         rel_logits = torch.cat((rel_logits_diagonal, rel_logits), dim=2)
#         rel_logits = torch.reshape(rel_logits, (B, V + 1, V))
#
#         # slice
#         flat_sliced = rel_logits[:, :V, :]
#         final_x = torch.reshape(flat_sliced, (B, V, V))
#         return final_x

import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


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


class SpatialAttention(nn.Module):
    def __init__(self, in_c, out_c, A, padding=0, kernel_size=1, shape=25, stride=1, t_dilation=1):
        super(SpatialAttention, self).__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.data_normalization = True
        self.skip_conn = True
        self.bn_flag = True
        self.drop_connect = True
        self.training = True
        self.dv = 0.25
        self.dk = 0.25
        self.Nh = 8
        self.num = 4
        self.num_point = shape
        self.num_subset = 3
        self.num_frame = 300
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.A = A[0] + A[1] + A[2]
        self.relative = True

        self.dk = int(out_c * self.dk)
        self.dv = int(out_c)

        if self.out_c == 64:
            self.frame = 200

        if self.out_c == 128:
            self.frame = 100

        if self.out_c == 256:
            self.frame = 50

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.pes = PositionalEncoding(self.in_c, self.num_point, self.num_frame, "spatial")
        self.qkv_conv = nn.Conv2d(self.in_c, 3 * self.dk + self.dv, kernel_size=self.kernel_size,
                                  stride=stride, padding=self.padding)
        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)


        # batch normalization
        self.data_bn = nn.BatchNorm1d(self.in_c * self.num_point)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, T, V = x.size()
        pos = self.pes(x)
        x_sum = x

        if self.data_normalization:
            x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
            x = self.data_bn(x)
            x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)

        # N, T, C, V > NT, C, 1, V
        xa = x.permute(0, 2, 1, 3).reshape(-1, C, 1, V)
        pos = pos.permute(0, 2, 1, 3).reshape(-1, C, 1, V)

        # Spatial Transformer
        B, _, Tt, Vv = xa.size()
        flat_q, flat_k, flat_g, flat_v, q, k, g, v = self.compute_flat_qkv(xa, self.dk, self.dv, self.Nh)
        po_q, po_k, po_v, po_g, q_po, k_po, g_po, v_po = self.position_projection(pos, self.dk, self.dv, self.Nh)

        mu_Q = flat_q.mean(-1).unsqueeze(-1)  # [1, D']
        mu_K = flat_k.mean(-1).unsqueeze(-1)  # [1, D']

        flat_q = flat_q - mu_Q
        flat_kq = flat_k - mu_K

        pairwise = torch.matmul(flat_q.transpose(2, 3), flat_k)
        unary = torch.matmul(mu_Q.transpose(2, 3), flat_g)
        logits_position = torch.matmul(po_q.transpose(2, 3), po_k)

        weights_pair = self.soft(pairwise + logits_position)
        weights_unary = self.soft(unary + logits_position)
        weights = torch.add(weights_pair, weights_unary)

        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (B, self.Nh, Tt, Vv, -1))
        attn_out = attn_out.permute(0, 1, 4, 2, 3)

        # combine_heads_2d, combine heads only after having calculated each Z separately
        # (batch, Nh*dv, 1, joints)
        attn_out = self.combine_heads_2d(attn_out)
        # Multiply for W0 (batch, out_channels, 1, joints) with out_channels=dv
        attn_out = self.attn_out(attn_out)

        # N, T, C, V > N, C, T, V
        attn_out = attn_out.reshape(N, T, -1, V).permute(0, 2, 1, 3)
        if self.skip_conn and self.in_c == self.out_c:
            y = attn_out + x_sum
        else:
            y = attn_out
        if self.bn_flag:
            y = self.bn(y)

        y = self.relu(y)

        return y

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        # T=1 in this case, because we are considering each frame separately
        N, _, T, V = qkv.size()

        q, k, g, v = torch.split(qkv, [dk, dk, dk, dv], dim=1)

        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        g = self.split_heads_2d(g, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q * (dkh ** -0.5)

        flat_q = torch.reshape(q, (N, Nh, dkh, T * V))
        flat_k = torch.reshape(k, (N, Nh, dkh, T * V))
        flat_g = torch.reshape(g, (N, Nh, dkh, T * V))
        flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, T * V))
        return flat_q, flat_k, flat_g, flat_v, q, k, g, v
    def position_projection (self, pos, dk, dv, Nh):
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
        B, channels, T, V = x.size()
        ret_shape = (B, Nh, channels // Nh, T, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, T, V = x.size()
        ret_shape = (batch, Nh * dv, T, V)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, T, V = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)
        q_first = q.unsqueeze(4).expand((B, Nh, T, V, V - 1, dk))
        q_first = torch.reshape(q_first, (B * Nh * T, -1, dk))

        # q used to multiply for the embedding of the parameter on the diagonal
        q = torch.reshape(q, (B * Nh * T, V, dk))
        # key_rel_diagonal: (1, dk) -> (V, dk)
        param_diagonal = self.key_rel_diagonal.expand((V, dk))
        rel_logits = self.relative_logits_1d(q_first, q, self.key_rel, param_diagonal, T, V, Nh)
        return rel_logits

    def relative_logits_1d(self, q_first, q, rel_k, param_diagonal, T, V, Nh):
        # compute relative logits along one dimension
        # (B*Nh*1,V^2-V, self.dk // Nh)*(V^2 - V, self.dk // Nh)

        # (B*Nh*1, V^2-V)
        rel_logits = torch.einsum('bmd,md->bm', q_first, rel_k)
        # (B*Nh*1, V)
        rel_logits_diagonal = torch.einsum('bmd,md->bm', q, param_diagonal)

        # reshapes to obtain Srel
        rel_logits = self.rel_to_abs(rel_logits, rel_logits_diagonal)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, V, V))
        return rel_logits

    def rel_to_abs(self, rel_logits, rel_logits_diagonal):
        B, L = rel_logits.size()
        B, V = rel_logits_diagonal.size()

        # (B, V-1, V) -> (B, V, V)
        rel_logits = torch.reshape(rel_logits, (B, V - 1, V))
        row_pad = torch.zeros(B, 1, V).to(rel_logits)
        rel_logits = torch.cat((rel_logits, row_pad), dim=1)

        # concat the other embedding on the left
        # (B, V, V) -> (B, V, V+1) -> (B, V+1, V)
        rel_logits_diagonal = torch.reshape(rel_logits_diagonal, (B, V, 1))
        rel_logits = torch.cat((rel_logits_diagonal, rel_logits), dim=2)
        rel_logits = torch.reshape(rel_logits, (B, V + 1, V))

        # slice
        flat_sliced = rel_logits[:, :V, :]
        final_x = torch.reshape(flat_sliced, (B, V, V))
        return final_x
