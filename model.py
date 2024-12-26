import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import mutual_info_score

class HONCD(nn.Module):
    def __init__(self, n_stu, n_pro, n_know, d0,  group, hid1, hid2, device):
        super(HONCD, self).__init__()

        self.stu = nn.Parameter(torch.rand(n_stu, d0))   
        self.theta = nn.Parameter(torch.rand(n_stu, n_know, d0))

        self.b = nn.Parameter(torch.rand(n_pro, n_know))
        self.disc = nn.Parameter(torch.rand(n_pro, 1))

        self.filter_layer = FilteringLayer(group, d0)
        self.interaction_layer = InteractionLayer(d0)
        self.aggregation_layer = AggregationLayer(d0)

        self.dt = nn.Parameter(torch.rand(1))
        self.know = nn.Parameter(torch.rand(n_stu, n_know))
        self.rt_features = nn.Parameter(torch.rand(151, n_know))
        self.rt_linear = nn.Linear(n_know, n_know)

        self.know_features = nn.Parameter(torch.rand(n_know, n_know))
        self.combine_layer = nn.Linear(3 * n_know, n_know)

        self.predictor = nn.Sequential(
            nn.Linear(n_know, hid1),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(hid1, hid2),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(hid2, 1),
            nn.Sigmoid()
        )

        for name, param in self.named_parameters():
            if 'weight' in name and len(param.size()) >= 2:
                nn.init.xavier_normal_(param)

    def forward(self, sid, pid, Q, rt, device, test=False):
        s_i = torch.sigmoid(self.stu[sid])
        theta_i = torch.sigmoid(self.theta[sid])

        theta_filter = self.filter_layer(s_i, theta_i, device)
        theta_interact = self.interaction_layer(theta_filter)
        theta_agg = self.aggregation_layer(s_i, theta_interact)

        b_j = torch.sigmoid(self.b[pid])
        disc_j = torch.sigmoid(self.disc[pid])

        rt_feature = self.rt_linear(torch.sigmoid(self.rt_features[rt]))
        know = torch.sigmoid(self.know[sid])
        x = torch.cat([rt_feature, know, theta_agg], dim=-1)
        x = disc_j * (self.combine_layer(x) - b_j) * Q

        output = self.predictor(x)

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        for module in self.predictor:
            module.apply(clipper)


    def forward(self, x):
        x = x.transpose(1, 2)  
        x = F.relu(self.conv1(x))
        return x

class FilteringLayer(nn.Module):
    '''
    group: array, start with 0
    '''
    def __init__(self, group, d0):
        super(FilteringLayer, self).__init__()
        self.group = group
        self.d0 = d0
        self.att = DotAttention(d0)

    def forward(self, s_i, theta_i, device):
        output = torch.empty(s_i.size()[0], len(self.group) - 1, self.d0).to(device)
        for i in range(1, len(self.group)):
            theta_group = theta_i[:, self.group[i - 1]:self.group[i], :]
            _, o = self.att(s_i.unsqueeze(1), theta_group, theta_group)
            output[:, i - 1, :] = o.squeeze(1)
        return output


class InteractionLayer(nn.Module):
    def __init__(self, d0):
        super(InteractionLayer, self).__init__()
        self.att = SelfAttention(d0, d0, d0)


    def forward(self, theta_filter):
        return self.att(theta_filter)


class AggregationLayer(nn.Module):
    def __init__(self, d0):
        super(AggregationLayer, self).__init__()
        self.att = DotAttention(d0)

    def forward(self, s_i, theta_interact):
        _, output = self.att(s_i.unsqueeze(1), theta_interact, theta_interact)
        return output.squeeze(1)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

class DotAttention(nn.Module):
    def __init__(self, scale):
        super(DotAttention, self).__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)
        self.attention_weights = None

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2))
        u = u / self.scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)

        attn = self.softmax(u)
        self.attention_weights = attn.detach().cpu().numpy()  # Save the attention weights
        output = torch.bmm(attn, v)
        return attn, output


class SelfAttention(nn.Module):
    def __init__(self, dim, dk, dv):
        super(SelfAttention, self).__init__()
        self.scale = dk ** -0.5
        self.q = nn.Linear(dim, dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim, dv)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        return x

