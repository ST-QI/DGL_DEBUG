from model import common
import torch
import torch.nn as nn
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos
import dgl


def make_model(options, parent=False):
    return Deblur_model(options)

class Deblur_model(nn.Module):
    def __init__(self, options, conv=common.default_conv):
        super(Deblur_model, self).__init__()
        #Typically, it needs 2 downsample layers for deblurring
        n_resblocks = int(options.getElementsByTagName('n_resblocks')[0].childNodes[0].nodeValue)  #number of resblocks is each stage
        n_filters = int(options.getElementsByTagName('n_filters')[0].childNodes[0].nodeValue)
        n_tokens = int(options.getElementsByTagName('n_tokens')[0].childNodes[0].nodeValue)
        n_graph_features = int(options.getElementsByTagName('n_graph_features')[0].childNodes[0].nodeValue)
        n_graph_layers = int(options.getElementsByTagName('n_graph_layers')[0].childNodes[0].nodeValue)
        kernel_size = 3
        act = nn.ReLU(True)
        n_ResGCN = int(options.getElementsByTagName('n_ResGCN')[0].childNodes[0].nodeValue)
        rgb_range = 255
        n_colors = 3

        self.cha_g, self.spa_g = None, None  # will be assigned value in __init__.py

        self.n_filters = n_filters
        self.sqrt_n_filters = int(np.sqrt(self.n_filters))
        self.n_tokens = n_tokens
        self.sqrt_n_tokens = int(np.sqrt(self.n_tokens))
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_filters, kernel_size)]

        # define body module 
        # encoder_1 -> encoder_2 -> encoder_3 -> GNN -> decoder_3 -> decoder_2 -> decoder_1
        encoder_1 = [common.ResBlock(conv, n_filters, 
                kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)]
        self.downsample_1 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=2, padding=1)

        encoder_2 = [common.ResBlock(conv, n_filters, 
                kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)]
        self.downsample_2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=2, padding=1)

        encoder_3 = [common.ResBlock(conv, n_filters, 
                kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)]
        self.downsample_graph = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=10, padding=1)

        GAT_body = [common.ChannelSpatialGATLayer(cha_nodes_n=self.n_filters,
                                                  cha_in_dim=n_tokens,
                                                  cha_out_dim=self.sqrt_n_tokens,
                                                  cha_num_heads=self.sqrt_n_tokens,
                                                  spa_nodes_n=self.n_tokens,
                                                  spa_in_dim=n_filters,
                                                  spa_out_dim=self.sqrt_n_filters,
                                                  spa_num_heads=self.sqrt_n_filters,
                                                  ) for _ in range(n_ResGCN)]
        self.graph_convhead = common.GraphConvolution(1, n_graph_features)
        self.graph_convtail = common.GraphConvolution(n_graph_features, 1)


        self.upsample_graph = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=kernel_size, stride=10, padding=1, output_padding=9)
        decoder_3 = [common.ResBlock(conv, n_filters, 
                kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)]

        self.upsample_2 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        decoder_2 = [common.ResBlock(conv, n_filters, 
                kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)]

        self.upsample_1 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        decoder_1 = [common.ResBlock(conv, n_filters, 
                kernel_size, act=act, res_scale=1) for _ in range(n_resblocks)]


        # define tail module
        m_tail = [conv(n_filters, n_colors, kernel_size)]

        self.relu = nn.LeakyReLU(0.2)
        self.head = nn.Sequential(*m_head)
        self.encoder_1 = nn.Sequential(*encoder_1)
        self.encoder_2 = nn.Sequential(*encoder_2)
        self.encoder_3 = nn.Sequential(*encoder_3)

        self.GAT_body = nn.Sequential(*GAT_body)

        #self.graph_conv = nn.Sequential(*graph_conv)
        self.decoder_3 = nn.Sequential(*decoder_3)
        self.decoder_2 = nn.Sequential(*decoder_2)
        self.decoder_1 = nn.Sequential(*decoder_1)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, input):
        x, cha_con, spa_con = input['x'], input['cha_con'], input['spa_con']
        #res = self.sub_mean(x)
        # x(b,3,240,240)
        res = self.head(x)  # res(b,64,240,240)

        res_enc1 = self.encoder_1(res)  # res_enc1(b,64,240,240)
        res_down1 = self.downsample_1(res_enc1)  # res_down1(b,64,120,120)
        res_enc2 = self.encoder_2(res_down1)  # res_enc2(b,64,120,120)
        res_down2 = self.downsample_2(res_enc2)  # res_down2(b,64,60,60)
        res_enc3 = self.encoder_3(res_down2)  # res_enc3(b,64,60,60)
        yin = self.downsample_graph(res_enc3)  # yin(b,64,6,6)

        yin = yin.permute(0,2,3,1)  # yin(b,6,6,64)
        yin = yin.unsqueeze(4)  # yin(b,6,6,64,1)
        yin = self.graph_convhead(yin)  # yin(b,6,6,64,36)

        # yin = yin.permute(3,0,1,2,4)  # yin(64,b,6,6,36)
        # res_g = self.GAT_body_channel(yin, device)  # res_g(64,b,6,6,36)
        # res_g = res_g.permute(4,1,2,3,0)  # res_g(36,b,6,6,64)
        # yout = self.GAT_body_spatial(res_g, device)  # res_g(36,b,6,6,64)
        # yout = yout.permute(1,2,3,4,0)  # res_g(b,6,6,64,36)

        yout, _, _ = self.GAT_body((yin, cha_con, spa_con))

        yout = self.graph_convtail(yout)  # res_g(b,6,6,64,1)
        yout = self.relu(yout)  # res_g(b,6,6,64,1)
        yout = yout.squeeze(4)  # res_g(b,6,6,64)
        yout = yout.permute(0,3,1,2)  # res_g(b,64,6,6)

        res_up3 = self.upsample_graph(yout)  # res_up3(b,64,60,60)
        res_dec3 = self.decoder_3(res_up3 + res_enc3)  # res_dec3(b,64,60,60)
        res_up2 = self.upsample_2(res_dec3)  # res_up2(b,64,120,120)
        res_dec2 = self.decoder_2(res_up2 + res_enc2)  # res_dec2(b,64,120,120)
        res_up1 = self.upsample_1(res_dec2)  # res_up1(b,64,240,240)
        res_out = self.decoder_1(res_up1 + res_enc1)  # res_out(b,64,240,240)

        res = self.tail(res_out)  # res(b,3,240,240)
        #res = self.add_mean(res)

        res += x  # res(b,3,240,240)
        return res

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
