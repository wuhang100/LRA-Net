#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:30:00 2020

@author: wuhang
"""

import sys
sys.path.append("../utils")
import tensorflow as tf
import tf_util
import layer_util
import numpy as np

class LAN ():
    def __init__(self, inputs,npts,cores,
                 num_cloud_exp_0,num_cloud_exp_1,
                 ball_index_0,ball_size_0,ball_index_1,ball_size_1,
                 nheads, reuse, in_drop, coef_drop, bn, train):
        
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * 1024
        self.global_features = self.global_encoder(inputs, npts, reuse)
        self.local_features = self.local_encoder(inputs,npts,reuse,cores,
                                                 num_cloud_exp_0,num_cloud_exp_1,
                                                 ball_index_0,ball_size_0,ball_index_1,ball_size_1)
        self.coarse, self.fine = self.gcn_decoder(self.global_features, self.local_features, cores, 
                                                  npts, nheads, reuse, in_drop, coef_drop, bn, train)

    def global_encoder(self, inputs, npts, reuse): 
        raw_fts,max_fts = layer_util.sharedMLP ('encoder_0',inputs, npts, [128,256], reuse)
        global_fts = tf_util.point_unpool(max_fts, npts) 
        features = tf.concat([raw_fts, global_fts], axis=2)
        
        _,features = layer_util.sharedMLP ('encoder_1',features, npts, [256,512], reuse)
        return features #[b, 1, 512]

    def local_encoder(self,inputs,npts,reuse,cores,
                      num_cloud_exp_0,num_cloud_exp_1,
                      ball_index_0,ball_size_0,ball_index_1,ball_size_1):
        batch_size = np.shape(npts)[0]
        cloud_exp_0 = tf.gather(inputs,ball_index_0,axis=1)
        _,max_fts_0 = layer_util.sharedMLP ('loc_encoder_0',cloud_exp_0, ball_size_0,[128,256], reuse)
        fts_exp_1 = tf.expand_dims(tf.gather(max_fts_0[:,0,:],ball_index_1),[0])
        _,max_fts_1 = layer_util.sharedMLP ('loc_encoder_1',fts_exp_1, ball_size_1,[512,512], reuse)
        loc_fts = tf.reshape(max_fts_1, [batch_size,cores[1],512])
        return loc_fts


    def gcn_decoder(self, glb_fts, loc_fts, cores, npts, nheads, reuse, in_drop, coef_drop, bn, train):
        batch_size = np.shape(npts)[0]
        head_pts = 1024/cores[1]/nheads
        points_set = []
        for i in range (nheads):
            points_set.append (layer_util.attnHead_all (str(i), loc_fts, glb_fts, 512, cores[1], tf.nn.relu, reuse,
                                                        train, in_drop=in_drop, coef_drop=coef_drop, bn_mode=bn))

        local_fts = tf.concat(points_set, axis=1)
        
        points = layer_util.sharedMLP_simple ('attn_decoder',local_fts, [256,256,head_pts*3], reuse, bn, train)
        points = tf.reshape(points,[batch_size, -1, head_pts, 3])
        coarse = tf.reshape(points,[batch_size, -1, 3])

	with tf.variable_scope ('decoder_fine', reuse=reuse):
	    coarse_fts = tf.reshape(coarse, [1, -1, 3]) 
            coarse_npts = 1024 * np.ones([batch_size],dtype=np.int32) 
	    _,coarse_fts = layer_util.sharedMLP ('coarse_feature',coarse_fts, coarse_npts,[128,256,512], reuse)
	    coarse_fts_exp = tf.tile(tf.expand_dims(coarse_fts, 2), [1,cores[1]*nheads,head_pts,1])

        x = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
        y = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
        grid = tf.meshgrid(x, y)
        grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
        grid_feat = tf.tile(grid, [batch_size, 1024, 1])

        local_fts_exp = tf.tile(tf.expand_dims(local_fts, 2), [1,1,head_pts,1])
        global_fts_exp = tf.tile(tf.expand_dims(glb_fts, 2), [1,cores[1]*nheads,head_pts,1])
        concat_fts = tf.concat([coarse_fts_exp,global_fts_exp],axis=3)
        concat_fts = tf.reshape(concat_fts,[batch_size, cores[1]*nheads*head_pts, 1024])

        couple_feat = tf.tile(tf.expand_dims(concat_fts, 2), [1, 1, self.grid_size ** 2, 1])
        couple_feat = tf.reshape(couple_feat, [batch_size, self.num_fine, 1024])

        point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        point_feat = tf.reshape(point_feat, [batch_size, self.num_fine, 3])

        feat = tf.concat([grid_feat, point_feat, couple_feat], axis=2)

        center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = tf.reshape(center, [batch_size, self.num_fine, 3])

        fine = layer_util.sharedMLP_simple('decoder_fine',feat, [512, 512, 256, 3], reuse, bn, train)+center
        return coarse, fine #[batch, 1024, 3],[batch, 1024*16, 3]
