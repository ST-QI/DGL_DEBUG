import os
import glob
import random
import pickle

from data import common
import cv2
import imageio
import torch
import torch.utils.data as data

import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos


class DBData(data.Dataset):
    def __init__(self, options, name='', train=True, benchmark=False):
        self.options = options
        self.name = name
        self.train = train
        self.do_eval = True
        self.idx_scale = 0
        self.dir_blur = []    #may be do not need
        self.dir_sharp = []
        self.dir_data = options.getElementsByTagName('dataroot')[0].childNodes[0].nodeValue
        self.patch_size = int(options.getElementsByTagName('patch_size')[0].childNodes[0].nodeValue)
        self.test_root = options.getElementsByTagName('test_root')[0].childNodes[0].nodeValue
        self.scale = 1 #There is no need to resize in image deblurring
        self.augment = False
        self.n_colors = 3
        self.rgb_range = 255
        self.batch_size = int(options.getElementsByTagName('batch_size')[0].childNodes[0].nodeValue)
        self.mode = options.getElementsByTagName('mode')[0].childNodes[0].nodeValue

        self.cha_adj_dir = options.getElementsByTagName('channel_adj_dir')[0].childNodes[0].nodeValue
        self.spa_adj_dir = options.getElementsByTagName('spatial_adj_dir')[0].childNodes[0].nodeValue

        list_hr, list_lr = self._scan()
        self.images_hr, self.images_lr = list_hr, list_lr
        #self.train = (self.mode == 'Train')
        if self.mode == 'Test':
            self._set_filesystem()

    # Below functions as used to prepare images
    def _scan(self):
        pass

    def _set_filesystem(self):
        pass      

    def __getitem__(self, idx):
        if self.mode == 'Train':
            lr, hr, filename = self._load_file(idx)
            # lr.shape = (720, 1280, 3)
            # hr.shape = (720, 1280, 3)
            pair = self.get_patch(lr, hr)
            pair = common.set_channel(*pair, n_channels=self.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.rgb_range)
            # pair_t[0].shape = (240, 240, 3)
            # pair_t[1].shape = (240, 240, 3)
            connectivity = self._load_graph()
            return pair_t[0], pair_t[1], filename, connectivity

        elif self.mode == 'Test':
            lr, _, filename = self._load_file(idx)
            pair = (lr, lr)
            pair = common.set_channel(*pair, n_channels=self.n_colors)
            lr_t = common.np2Tensor(*pair, rgb_range=self.rgb_range)
            connectivity = self._load_graph()
            return lr_t[0], 0, filename, connectivity

    def __len__(self):
        return len(self.images_lr)#have changed for deblurring

    def _load_file(self, idx):
        hr = 0
        f_lr = self.images_lr[idx]
        if self.mode == 'Train':
            f_hr = self.images_hr[idx]
            hr = imageio.imread(f_hr)
            filename, _ = os.path.splitext(os.path.basename(f_lr))
        elif self.mode == 'Test':
            filename = f_lr.replace(self.dir_data + '/GoPro_test', self.test_root)

        lr = imageio.imread(f_lr)

        return lr, hr, filename

    def _load_graph(self):
        cha_adj = self.get_Adj(adj_file=self.cha_adj_dir)
        spa_adj = self.get_Adj(adj_file=self.spa_adj_dir)
        cha_con = algos.adj2graph(cha_adj)
        spa_con = algos.adj2graph(spa_adj)
        return cha_con, spa_con
        # return {'cha_con': cha_con, 'spa_con': spa_con}

    def get_patch(self, lr, hr):
        scale = self.scale
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.patch_size,
                scale=scale,
                multi=False,
                input_large=False
            )
            if self.augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    @staticmethod
    def get_Adj(adj_file):
        import scipy.io as spio
        data = spio.loadmat(adj_file)
        data = data['FULL'].astype(np.int64)
        return data
