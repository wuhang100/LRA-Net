#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:02:06 2019

@author: wuhang
"""

import tensorflow as tf
import os
import numpy as np
import sys
import heapq
import pprint
sys.path.append("../")
from pc_distance import tf_nndistance, tf_approxmatch

def dense (scope_name,
           inputs, 
           output_num,  
           is_training, 
           reuse_mode = False, 
           activation_fun = None,
           regularizer = tf.contrib.layers.l2_regularizer(1e-6),
           uniform_initializer = True,
           use_bn = False,
           bn_momentum = 0.9):
    with tf.variable_scope (scope_name, reuse=reuse_mode):      
        initializer = tf.contrib.layers.xavier_initializer(uniform = uniform_initializer)            
        outputs = tf.layers.dense(inputs, output_num,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer)
        if use_bn == True:            
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_momentum, training=is_training)            
        if activation_fun is not None:
            outputs = activation_fun(outputs)
    return outputs

def conv1d (scope_name,
           inputs, 
           output_channel,  
           is_training, 
           reuse_mode = False, 
           activation_fun = None,
           regularizer = tf.contrib.layers.l2_regularizer(1e-6),
           uniform_initializer = True,
           use_bn = False,
           bn_momentum = 0.9):
    with tf.variable_scope (scope_name, reuse=reuse_mode):
        initializer = tf.contrib.layers.xavier_initializer(uniform = uniform_initializer)        
        outputs = tf.layers.conv1d(inputs, output_channel, kernel_size=1, 
                                   kernel_initializer=initializer,
                                   kernel_regularizer=regularizer)
        if use_bn == True:            
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_momentum, training=is_training)        
        if activation_fun is not None:
            outputs = activation_fun(outputs)
    return outputs

# this function is borrowed from PCN
def point_maxpool(inputs, npts, keepdims=False):
    outputs = [tf.reduce_max(f, axis=1, keepdims=keepdims)
        for f in tf.split(inputs, npts, axis=1)]
    return tf.concat(outputs, axis=0)

# this function is borrowed from PCN
def point_unpool(inputs, npts):
    inputs = tf.split(inputs, inputs.shape[0], axis=0)
    outputs = [tf.tile(f, [1, npts[i], 1]) for i,f in enumerate(inputs)]
    return tf.concat(outputs, axis=1)

def sort_path (input_path,output_path1,output_path2, num_view):
    input0 = [os.path.join(input_path,i) for i in os.listdir(input_path)]
    output1 = [os.path.join(output_path1,i) for i in os.listdir(output_path1)]
    output2 = [os.path.join(output_path2,i) for i in os.listdir(output_path2)]
    output1 = np.array(output1)
    output1 = np.reshape([np.tile(output1[i],num_view) for i in range (np.shape(output1)[0])],[-1])
    output1 = output1.tolist()
    output2 = np.array(output2)    
    output2 = np.reshape([np.tile(output2[i],num_view) for i in range (np.shape(output2)[0])],[-1])
    output2 = output2.tolist()
    return input0,output1,output2

def prepare_2out (input_path,output_path1,output_path2,batch_size, num_view,shuffle=True):
    input0,output1,output2 = sort_path (input_path,output_path1,output_path2, num_view)
    input0.sort()
    output1.sort()
    output2.sort()
    dataset = tf.data.Dataset.from_tensor_slices((input0,output1,output2))
    if (shuffle==True):
        dataset = dataset.shuffle(buffer_size=len(input0))
    dataset = dataset.batch(batch_size)
    return dataset,len(input0)

def printvar (tvars):
    pp = pprint.PrettyPrinter()
    pp.pprint(tvars)

# this function is borrowed from PCN
def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return (dist1 + dist2) / 2

# this function is borrowed from PCN
def earth_mover(pcd1, pcd2):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return tf.reduce_mean(cost / num_points)