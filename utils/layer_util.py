#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:42:19 2020

@author: wuhang
"""

import sys
import tf_util
import tensorflow as tf
import numpy as np

def MLP (scope_name,features, layer_dims, reuse_mode, bn_mode=False, train_mode=True, 
         reg = tf.contrib.layers.l2_regularizer(1e-3)):
    with tf.variable_scope (scope_name, reuse=reuse_mode):
        for i, num_outputs in enumerate(layer_dims[:-1]):
            features = tf_util.dense('fc'+str(i),features,num_outputs,
                                     train_mode,reuse_mode,
                                     tf.nn.relu,use_bn=bn_mode,regularizer=reg)
        outputs = tf_util.dense('fc'+str(i+1), features, layer_dims[-1],train_mode,reuse_mode,
                                use_bn=bn_mode,regularizer=reg)
    return outputs

def sharedMLP (scope_name,inputs, npts, layer_dims, reuse_mode, bn_mode=False, train_mode=True,
               reg = tf.contrib.layers.l2_regularizer(1e-3)):
    """
    inputs:[1,None,3],npts:[batch_size]
    Features: [batch_size, 1, layer_dims[-1]], outputs: [1, None, layer_dims[-1]]
    """
    with tf.variable_scope (scope_name, reuse=reuse_mode):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf_util.conv1d('sfc'+str(i), inputs, num_out_channel, 
                                    train_mode, reuse_mode,
                                    tf.nn.relu, use_bn = bn_mode,
                                    regularizer=reg)
        outputs = tf_util.conv1d('sfc'+str(len(layer_dims)-1), inputs, layer_dims[-1],train_mode,reuse_mode,
                                 use_bn=bn_mode,regularizer=reg)
        features = tf_util.point_maxpool(outputs, npts, keepdims=True)
    return outputs,features

def sharedMLP_simple (scope_name,inputs, layer_dims, reuse_mode, bn_mode=False, train_mode=True,
                      reg = tf.contrib.layers.l2_regularizer(1e-3)):
    """
    inputs:[1,None,3],npts:[batch_size]
    Features: [batch_size, 1, layer_dims[-1]], outputs: [1, None, layer_dims[-1]]
    """
    with tf.variable_scope (scope_name, reuse=reuse_mode):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf_util.conv1d('sfc'+str(i), inputs, num_out_channel, 
                                    train_mode, reuse_mode,
                                    tf.nn.relu, use_bn = bn_mode,
                                    regularizer=reg)
        outputs = tf_util.conv1d('sfc'+str(len(layer_dims)-1), inputs, layer_dims[-1],train_mode,reuse_mode,
                                 use_bn=bn_mode,regularizer=reg)
    return outputs

def attnHead_all (hid, local_feature, global_feature, out_size, num_sample, activation, reuse_mode, train_mode,
                  in_drop=0.0, coef_drop=0.0, bn_mode=False, reg = tf.contrib.layers.l2_regularizer(1e-3)):
    """
    local_feature: [batch_size, cores[1], 512], global_feature: [b, 1, 512]
    hid: head_id
    Returns: #[b, num_sample, out_size]
    """

    global_fts = tf.tile(global_feature,[1,num_sample,1])      #[batch_size, num_sample, 512]
    concat_fts = tf.concat([global_fts, local_feature],axis=2)  #[batch_size, num_sample, 512+512]    
    name = 'attn'+str(hid)
    with tf.variable_scope(name, reuse=reuse_mode):
        if in_drop != 0.0:
            concat_fts = tf.layers.dropout(concat_fts, rate=in_drop, training=train_mode)
        
        #[batch_size, num_sample, out_size]
	seq_fts = sharedMLP_simple ('fts_l1',concat_fts, [out_size*2,out_size], reuse_mode, bn_mode, train_mode)

        f_1 = sharedMLP_simple ('f_1',seq_fts, [out_size/2,1], reuse_mode, bn_mode, train_mode)   #[batch_size, num_sample, 1]
        f_2 = sharedMLP_simple ('f_2',seq_fts, [out_size/2,1], reuse_mode, bn_mode, train_mode)   #[batch_size, num_sample, 1]
        logits = f_1 + tf.transpose(f_2, [0,2,1])   	    #[batch_size, num_sample, num_sample]
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))     #[batch_size, num_sample, num_sample]

        if coef_drop != 0.0:
            coefs = tf.layers.dropout(coefs, rate=coef_drop, training=train_mode)
        if in_drop != 0.0:
            seq_fts = tf.layers.dropout(seq_fts, rate=in_drop, training=train_mode)

        vals = tf.matmul(coefs, seq_fts)  #[batch_size, num_sample, out_size]
        vals_wres = vals + sharedMLP_simple ('res',concat_fts, [out_size], reuse_mode, bn_mode, train_mode)  # residual connect 
        return activation(vals_wres) #[batch_size, num_sample, out_size]
