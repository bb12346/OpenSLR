import torch
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# import sys
# sys.path.append('/root/SSL/OpenSLR/opengait/modeling')
# from base_model import BaseModel
# from modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks


class i3d_mlp_inference(nn.Module):
    def __init__(self):
        super(i3d_mlp_inference, self).__init__()

    def build_network(self, model_cfg):
        # if model_cfg['i3d_pretrained']:
        #     self.i3d = model_cfg['i3d_pretrained']
        # else:
        self.i3d = InceptionI3d(
            num_classes=60,
            spatiotemporal_squeeze=True,
            final_endpoint="Logits",
            name="inception_i3d",
            in_channels=3,
            dropout_keep_prob=0.5,
            num_in_frames=16,
            include_embds=True,
            model_cfg = model_cfg
        )
        # if model_cfg['mlp_pretrained']:
        #     self.mlp = model_cfg['mlp_pretrained']
        # else:
        # self.mlp = Mlp()

    def forward(self, inputs):
        # ipts, labs, _, _, seqL = inputs
        # seqL = None if not self.training else seqL
        # if not self.training and len(labs) != 1:
        #     raise ValueError(
        #         'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        # # sils = ipts[0].permute(0,2,1,3,4)
        # # print('-sils-',sils.shape)
        # del ipts
        # n, _, s, h, w = sils.size()
        # if s < 3:
        #     repeat = 3 if s == 1 else 2
        #     sils = sils.repeat(1, 1, repeat, 1, 1)

        i3d_outputs = self.i3d(inputs)
        # logits from i3d
        logits = i3d_outputs["logits"] #[27,60]
        logits = logits.unsqueeze(1) #[27, 1, 60]
        # [B, 1024, 1, 1, 1] => [B, 1024]
        x = i3d_outputs["embds"].squeeze(2).squeeze(2).squeeze(2)# [27, 1024]

        # Get embds from mlp (unused logits from mlp)
        # embds = self.mlp(x)["embds"] # [27,512]

        # n, _, s, h, w = sils.size()
        retval = {
            # 'training_feat': {
            #     # 'triplet': {'embeddings': embds, 'labels': labs},
            #     'softmax': {'logits': logits, 'labels': labs}
            # },
            # 'visual_summary': {
            #     # 'image/sils': sils.view(n*s, 1, h, w)
            #     'image/sils': sils
            # },
            'inference_feat': {
                'embeddings': x
            }
        }
        return retval

        # return {"logits": logits, "embds": embds}
        # return i3d_outputs

class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,ms.shaw
        # out_t = np.ceil(float(t) / float(self.stride[0]))
        # out_h = np.ceil(float(h) / float(self.stride[1]))
        # out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name="unit_3d",
        num_domains=1,
    ):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._num_domains = num_domains
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,  # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
            bias=self._use_bias,
        )

        if self._use_batch_norm:
            if self._num_domains == 1:
                self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)
            else:
                self.bn = DomainSpecificBatchNorm3d(
                    self._output_channels, self._num_domains, eps=0.001, momentum=0.01
                )

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        # out_t = np.ceil(float(t) / float(self._stride[0]))
        # out_h = np.ceil(float(h) / float(self._stride[1]))
        # out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name, num_domains=1):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0
        )
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)

def get_inplanes():
    return [64, 128, 256, 1024]

class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        num_classes=60,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=64,
        include_embds=False,
        model_cfg = None
    ):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatiotemporal_squeeze: Whether to squeeze the 2 spatial and 1 temporal dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          in_channels: Number of input channels (default 3 for RGB).
          dropout_keep_prob: Dropout probability (default 0.5).
          name: A string (optional). The name of this module.
          num_in_frames: Number of input frames (default 64).
          include_embds: Whether to return embeddings (default False).
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError("Unknown final endpoint %s" % final_endpoint)

        super().__init__()
        self._num_classes = num_classes
        self._spatiotemporal_squeeze = spatiotemporal_squeeze
        self._final_endpoint = final_endpoint
        self.include_embds = include_embds
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError("Unknown final endpoint %s" % self._final_endpoint)

        self.end_points = {}
        end_point = "Conv3d_1a_7x7"
        self.end_points[end_point] = Unit3D(
            in_channels=in_channels,
            output_channels=64,
            kernel_shape=[7, 7, 7],
            stride=(2, 2, 2),
            padding=3,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_2a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2b_1x1"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=64,
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=192,
            kernel_shape=[3, 3, 3],
            padding=1,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_3a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3b"
        self.end_points[end_point] = InceptionModule(
            192, [64, 96, 128, 16, 32, 32], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionModule(
            256, [128, 128, 192, 32, 96, 64], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_4a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4b"
        self.end_points[end_point] = InceptionModule(
            128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4c"
        self.end_points[end_point] = InceptionModule(
            192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionModule(
            160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionModule(
            128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4f"
        self.end_points[end_point] = InceptionModule(
            112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_5a_2x2"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5b"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5c"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Logits"

        last_duration = int(math.ceil(num_in_frames / 8))  # 8
        last_size = 7  # int(math.ceil(sample_width / 32))  # this is for 224
        # self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.dropout = nn.Dropout(dropout_keep_prob)

        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

        self.build()


    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, sils):
        x = sils
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        # [batch x featuredim x 1 x 1 x 1]
        embds = self.dropout(self.avgpool(x)) # torch.Size([27, 1024, 1, 1, 1])

        # [batch x classes x 1 x 1 x 1]
        x = self.logits(embds)  # torch.Size([27, 60, 1, 1, 1])
        if self._spatiotemporal_squeeze:
            # [batch x classes]
            logits = x.squeeze(3).squeeze(3).squeeze(2)

        # logits [batch X classes]
        if self.include_embds:
            return {"logits": logits, "embds": embds}
        else:
            return {"logits": logits}

        #
        #
        # n, _, s, h, w = sils.size()
        # retval = {
        #     'training_feat': {
        #         'triplet': {'embeddings': embed_1, 'labels': labs},
        #         'softmax': {'logits': logits, 'labels': labs}
        #     },
        #     'visual_summary': {
        #         # 'image/sils': sils.view(n*s, 1, h, w)
        #         'image/sils': sils
        #     },
        #     'inference_feat': {
        #         'embeddings': embed
        #     }
        # }
        # return retval

        # [batch x classes x 1 x 1 x 1]

class Mlp(torch.nn.Module):
    def __init__(
        self,
        input_dim=1024,
        adaptation_dims=[512, 256],
        # adaptation_dims=[512],
        with_norm=True,
        num_classes=60,
        with_classification=False,
        dropout_keep_prob=0.0,
    ):
        super(Mlp, self).__init__()
        self.input_dim = input_dim
        self.with_norm = with_norm
        self.with_classification = with_classification
        self.dropout = nn.Dropout(dropout_keep_prob)

        # 1. Layers learning a 1024d residual
        self.res = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.LeakyReLU(0.2, inplace=True)
        )
        # 2. Layers learning the embeddings from 1024d -> 256d
        layers = []
        layers.append(nn.Linear(input_dim, adaptation_dims[0]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(1, len(adaptation_dims)):
            layers.append(nn.Linear(adaptation_dims[i - 1], adaptation_dims[i]))
            # layers.append(nn.BatchNorm1d(adaptation_dims[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.adaptor = nn.Sequential(*layers)
        output_dim = adaptation_dims[-1]
        # 3. Classifier
        self.logits = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        x = x + self.res(x)
        x = self.adaptor(x)
        # L2 normalize each feature vector
        if self.with_norm:
            x = F.normalize(x, p=2, dim=1)
        out = {"embds": x}
        if self.with_classification:
            logits = self.logits(self.dropout(x))
            out["logits"] = logits
        return out