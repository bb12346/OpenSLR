import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import warnings
# import sys
# sys.path.append('/root/SSL/OpenSLR/opengait/modeling')
# from base_model import BaseModel
# from modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks

class DropBlockT_1d(nn.Module):
    def __init__(self, block_size=7):
        super(DropBlockT_1d, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size

    def forward(self, input, keep_prob):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n,c,t,v = input.size()

        input_abs = torch.mean(torch.mean(torch.abs(input),dim=3),dim=1).detach()
        input_abs = (input_abs/torch.sum(input_abs)*input_abs.numel()).view(n,1,t)
        gamma = (1. - self.keep_prob) / self.block_size
        input1 = input.permute(0,1,3,2).contiguous().view(n,c*v,t)
        M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1,c*v,1)
        Msum = F.max_pool1d(M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2)
        mask = (1 - Msum).to(device=input.device, dtype=input.dtype)
        return (input1 * mask * mask.numel() /mask.sum()).view(n,c,v,t).permute(0,1,3,2)

class DropBlock_Ske(nn.Module):
    def __init__(self, num_point, block_size=7):
        super(DropBlock_Ske, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size
        self.num_point = num_point

    def forward(self, input, keep_prob, A):  # n,c,t,v
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.size()

        input_abs = torch.mean(torch.mean(
            torch.abs(input), dim=2), dim=1).detach()
        input_abs = input_abs / torch.sum(input_abs) * input_abs.numel()
        if self.num_point == 25:  # Kinect V2
            gamma = (1. - self.keep_prob) / (1 + 1.92)
        elif self.num_point == 20:  # Kinect V1
            gamma = (1. - self.keep_prob) / (1 + 1.9)
        else:
            gamma = (1. - self.keep_prob) / (1 + 1.92)
            # warnings.warn('undefined skeleton graph')
        M_seed = torch.bernoulli(torch.clamp(
            input_abs * gamma, max=1.0)).to(device=input.device, dtype=input.dtype)
        M = torch.matmul(M_seed, A)
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0
        mask = (1 - M).view(n, 1, 1, self.num_point)
        return input * mask * mask.numel() / mask.sum()

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

class Graphx:
    def __init__(self, labeling_mode='spatial'):
        #num_node = 27 
        num_node = 39 # 9 + 15 +15 (angel-axis)
        self_link = [(i, i) for i in range(num_node)]
        """
        inward_ori_index = [(5, 6), (5, 7),
                            (6, 8), (8, 10), (7, 9), (9, 11), 
                            (12,13),(12,14),(12,16),(12,18),(12,20),
                            (14,15),(16,17),(18,19),(20,21),
                            (22,23),(22,24),(22,26),(22,28),(22,30),
                            (24,25),(26,27),(28,29),(30,31),
                            (10,12),(11,22)]
        inward = [(i - 5, j - 5) for (i, j) in inward_ori_index]
        """
        inward_ori_index = [(1, 2), (1, 6),
                            (3, 2), (4, 3), (5, 4), 
                            (7, 6), (8, 7), (9, 8),
                            (12,11),(11,10),(10,5),
                            (15,14),(14,13),(13,5),
                            (18,17),(17,16),(16,5),
                            (21,20),(20,19),(19,5),
                            (24,23),(23,22),(22,5),
                            (27,26),(26,25),(25,9),
                            (30,29),(29,28),(28,9),
                            (33,32),(32,31),(31,9),
                            (36,35),(35,34),(34,9),
                            (39,38),(38,37),(37,9)]
        
        inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(self.num_node, self.self_link, self.inward, self.outward)

    def get_adjacency_matrix(self,num_node,self_link,inward,outward,labeling_mode='spatial'):

        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            print('-labeling_mode 1-')
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2)))
    nn.init.constant(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, num_point=25, block_size=41):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob, A):
        x = self.bn(self.conv(x))
        x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class unit_tcn_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn_skip, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
                                      3, 1, num_point, num_point]), dtype=torch.float32, requires_grad=True).repeat(1, groups, 1, 1), requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * num_subset)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(
            in_channels, out_channels * num_subset, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * num_subset)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * num_subset, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant_(self.Linear_bias, 1e-6)

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(num_point))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(
            eye_array), requires_grad=False, device='cuda'), requires_grad=False)  # [c,25,25]

    def norm(self, A):
        b, c, h, w = A.size()
        A = A.view(c, self.num_point, self.num_point)
        D_list = torch.sum(A, 1).view(c, 1, self.num_point)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0):
        learn_A = self.DecoupleA.repeat(
            1, self.out_channels // self.groups, 1, 1)
        norm_learn_A = torch.cat([self.norm(learn_A[0:1, ...]), self.norm(
            learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])], 0)

        x = torch.einsum(
            'nctw,cd->ndtw', (x0, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias
        x = self.bn0(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum('nkctv,kcvw->nctw', (x, norm_learn_A))

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, block_size, stride=1, residual=True, attention=True):
        super(TCN_GCN_unit, self).__init__()
        num_jpts = A.shape[-1]
        self.gcn1 = unit_gcn(in_channels, out_channels, A, groups, num_point)
        self.tcn1 = unit_tcn(out_channels, out_channels,
                             stride=stride, num_point=num_point)
        self.relu = nn.ReLU()

        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
                              3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'), requires_grad=False)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_skip(
                in_channels, out_channels, kernel_size=1, stride=stride)
        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)
        self.attention = attention
        if attention:
            #print('Attention Enabled!')
            self.sigmoid = nn.Sigmoid()
            # temporal attention
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)
            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x, keep_prob):
        y = self.gcn1(x)
        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)
            
        y = self.tcn1(y, keep_prob, self.A)
        x_skip = self.dropT_skip(self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        return self.relu(y + x_skip)


    # def build_network(self, model_cfg):

class SLR_pose3D_inference(nn.Module):
    def __init__(self):
        super().__init__()

    def build_network(self, model_cfg):
    # def build_network(self,num_class=60):
        num_class = model_cfg['class_num']
        # num_class = 60
        num_person=1
        num_point=39
        groups=16
        block_size=41 
        # graph=Graphx()
        graph_args={'labeling_mode': 'spatial'}
        in_channels=3
        # if graph is None:
            # raise ValueError()
        # else:
            # Graph = import_class(graph)
        self.graph = Graphx(**graph_args)
        # channels = [64, 128, 256]
        channels = [64, 128, 256]

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, channels[0], A, groups, num_point,
                               block_size, residual=False)
        self.l2 = TCN_GCN_unit(channels[0], channels[0], A, groups, num_point, block_size)
        self.l3 = TCN_GCN_unit(channels[0], channels[0], A, groups, num_point, block_size)
        self.l4 = TCN_GCN_unit(channels[0], channels[0], A, groups, num_point, block_size)
        self.l5 = TCN_GCN_unit(channels[0], channels[1], A, groups, num_point, block_size, stride=1)
        self.l6 = TCN_GCN_unit(channels[1], channels[1], A, groups, num_point, block_size)
        self.l7 = TCN_GCN_unit(channels[1], channels[1], A, groups, num_point, block_size)
        self.l8 = TCN_GCN_unit(channels[1], channels[2], A, groups, num_point, block_size, stride=1)
        self.l9 = TCN_GCN_unit(channels[2], channels[2], A, groups, num_point, block_size)
        self.l10 =TCN_GCN_unit(channels[2], channels[2], A, groups, num_point, block_size)

        # self.fc = nn.Linear(256, num_class)
        # nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        
        self.Head0 = SeparateFCs(39, channels[2], channels[2])


        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])


    def forward(self, inputs):
        keep_prob=0.9
        sils = inputs
        # print('-x1-',sils.shape)
        N, C, T, V, M = sils.size()
        x = sils.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # print('-x2-',x.shape)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        # print('-x3-',x.shape)
        x = self.l1(x, 1.0)
        x = self.l2(x, 1.0)
        x = self.l3(x, 1.0)
        x = self.l4(x, 1.0)
        x = self.l5(x, 1.0)
        x = self.l6(x, 1.0)
        xl7 = self.l7(x, keep_prob)
        xl8 = self.l8(xl7, keep_prob)
        xl9 = self.l9(xl8, keep_prob)
        xl10 = self.l10(xl9, keep_prob)
        # print('-l10-',x.shape)

        # N*M,C,T,V
        # c_new = x.size(1)

        # print(x.shape)
        # print(N, M, c_new)

        # x = x.view(N, M, c_new, -1)
        # x = x.reshape(N, M, c_new, -1)
        # print('-reshape-',x.shape)
        # x = x.mean(3).mean(1)
        xl10 = xl10.mean(2)
        # print('-xl10 -',xl10.shape)
        xl9 = xl9.mean(2)
        # print('-xl9 -',xl9.shape)
        #xl8 = xl8.mean(2)
        # print('-xl8 -',xl8.shape)
        x = torch.cat((xl9, xl10), dim=2)
        # outs = x.unsqueeze(1)
        # print('-outs1-',outs.shape)
        # outs = outs.permute(1, 0, 2).contiguous()  # [p, n, c]
        outs = xl10.permute(2, 0, 1).contiguous()  # [p, n, c]
        # print('-outs2-',outs.shape)


        embed_1 = self.Head0(outs)  # [p, n, c]
        # print('-Head0-',outs.shape)
        # gait = gait.permute(1, 2, 0).contiguous()  # [n, c, p]
        # gait = gait.permute(0, 2, 1).contiguous()  # [n, p, c]



        # feat = self.HPP(outs)  # [n, c, p]
        # feat = feat.permute(2, 0, 1).contiguous()  # [p, n, c]

        # embed_1 = self.FCs(feat)  # [p, n, c]
        embed_2, logits = self.BNNecks(embed_1)  # [p, n, c]

        embed_1 = embed_1.permute(1, 0, 2).contiguous()  # [n, p, c]
        embed_2 = embed_2.permute(1, 0, 2).contiguous()  # [n, p, c]
        logits = logits.permute(1, 0, 2).contiguous()  # [n, p, c]
        embed = embed_1
        # print('-embed-',embed.shape)

        n, _, s, h, w = sils.size()
        retval = {
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
# if __name__ == "__main__":
#     print('-------1------')
#     model = SLR_Decouple()
#     model.build_network()
#     model.cuda()
#     # model = SLR_Decouple()
#     # model.build_network(cfgs['model_cfg'])
#     x = torch.ones(10 * 3 * 16 * 27 * 1).reshape(10, 3, 16, 27, 1).cuda()
#     print('x=', x.shape)
#     a = model(x)
#     print('a=',a.shape)