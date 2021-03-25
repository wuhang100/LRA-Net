#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:57:16 2020

@author: wuhang
"""

import tensorflow as tf
import argparse
import numpy as np
import io
import sys
#import open3d as o3d
sys.path.append("./utils")
import tf_util
import np_util
sys.path.append("./models")
import model
import csv

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=30)
parser.add_argument('-num_repeat', type=int, default=80)

parser.add_argument('-num_view', type=int, default=30)
parser.add_argument('-train_path', default='../data/modelnet_train/partial')
parser.add_argument('-coarse_path', default='../data/modelnet_train/full_1k')
parser.add_argument('-fine_path', default='../data/modelnet_train/full_1w')
parser.add_argument('-valid_path', default='../data/modelnet_test/partial')
parser.add_argument('-valid_coarse_path', default='../data/modelnet_test/full_1k')
parser.add_argument('-valid_fine_path', default='../data/modelnet_test/full_1w')

parser.add_argument('-log_path', default='./log/')
parser.add_argument('-save_path', default='./restore/fine/intest')
parser.add_argument('-restore', type=bool, default=False)
parser.add_argument('-checkpoint', default='./restore/fine/')

parser.add_argument('-cores', type=int, default=[64,16])  
parser.add_argument('-min_dist', type=float, default=[0.25,0.5])
parser.add_argument('-min_points', type=int, default=[8,0])
parser.add_argument('-nheads', type=int, default=16)

parser.add_argument('-test', action='store_true', default=False)
parser.add_argument('-base_lr', type=float, default=0.0001)
parser.add_argument('-lr_decay', action='store_true')
parser.add_argument('-lr_decay_steps', type=int, default=50000)
parser.add_argument('-lr_decay_rate', type=float, default=0.7)
parser.add_argument('-lr_clip', type=float, default=1e-6)

parser.add_argument('-bn', type=bool, default=False)
parser.add_argument('-in_drop', type=float, default=0.0)
parser.add_argument('-coef_drop', type=float, default=0.0)
parser.add_argument('-reg_coef', type=float, default=0.1)

parser.add_argument('-print_list', action='store_true', default=False)

args = parser.parse_args()


if (args.test==True):
    batch_size = 1
else:
    batch_size = args.batch_size

""" 
Training parameters
"""
#global_step = tf.Variable(0, trainable=False, name='global_step')
global_step = tf.Variable(0, trainable=False, name='global_step')
alpha = tf.train.piecewise_constant(global_step, [5000, 1000, 30000],[0.01, 0.1, 0.5, 1.0], 'alpha_op')

min_loss_coarse = 0.5
min_loss_fine = 0.5
train_loss = tf.placeholder(tf.float32, shape=[])
train_loss_coarse = tf.placeholder(tf.float32, shape=[])
train_loss_fine = tf.placeholder(tf.float32, shape=[])
valid_loss = tf.placeholder(tf.float32, shape=[])
valid_loss_coarse = tf.placeholder(tf.float32, shape=[])
valid_loss_fine = tf.placeholder(tf.float32, shape=[])
"""
Model parameters
"""
inputs_pl = tf.placeholder(tf.float32, [1, None, 3], 'inputs')
npts_pl = tf.placeholder(tf.int32, [batch_size,], 'num_points')
gt_pl = tf.placeholder(tf.float32, [batch_size, 1024, 3], 'ground_truths')
gt_fine_pl = tf.placeholder(tf.float32, [batch_size, 16384, 3], 'ground_truth_fine')

num_cloud_exp_0_pl = tf.placeholder(tf.int32, shape=[batch_size])
num_cloud_exp_1_pl = tf.placeholder(tf.int32, shape=[batch_size])

ball_index_0_pl = tf.placeholder(tf.int32, shape=[None])
ball_size_0_pl = tf.placeholder(tf.int32, shape=[batch_size*args.cores[0]])

ball_index_1_pl = tf.placeholder(tf.int32, shape=[None])
ball_size_1_pl = tf.placeholder(tf.int32, shape=[batch_size*args.cores[1]])

train_pl = tf.placeholder(tf.bool, shape=[])

if args.lr_decay:
    learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                               args.lr_decay_steps, args.lr_decay_rate,
                                               staircase=True, name='lr')
    learning_rate = tf.maximum(learning_rate, args.lr_clip)
else:
    learning_rate = tf.constant(args.base_lr, name='lr')


def get_loss(coarse, fine, gt, gt_fine, reg_coef, is_training):
    
    reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_coarse = [reg_c for reg_c in reg if not 'decoder_fine' in reg_c.name]
    reg_fine = [reg_f for reg_f in reg if 'decoder_fine' in reg_f.name]

    loss_reg_coarse = reg_coef * tf.cast(is_training,tf.float32)*tf.reduce_mean(reg_coarse)
    loss_reg_fine = 0.25 * reg_coef * tf.cast(is_training,tf.float32)*tf.reduce_mean(reg_fine)

    loss_coarse = tf_util.earth_mover(coarse,gt) + loss_reg_coarse
    loss_fine = tf_util.chamfer(fine, gt_fine) + loss_reg_fine

    return loss_coarse, loss_reg_coarse, loss_fine, loss_reg_fine

MODEL = model.LAN(inputs_pl,npts_pl,args.cores,
                  num_cloud_exp_0_pl,num_cloud_exp_1_pl,
                  ball_index_0_pl, ball_size_0_pl,ball_index_1_pl, ball_size_1_pl,
                  args.nheads, False, args.in_drop, args.coef_drop, args.bn, train_pl)

pc_coarse = MODEL.coarse
pc_fine = MODEL.fine
loss_coarse,loss_reg_coarse,loss_fine,loss_reg_fine = get_loss(pc_coarse, pc_fine, gt_pl, gt_fine_pl, args.reg_coef, train_pl)

tvars = tf.trainable_variables()
#tf_util.printvar (tvars)
c_vars = [var for var in tvars if not 'decoder_fine' in var.name]
f_vars = [var for var in tvars if 'decoder_fine' in var.name]
tf_util.printvar (f_vars)

train_coarse = tf.train.AdamOptimizer(learning_rate).minimize(loss_coarse,var_list=c_vars,global_step=global_step)
train_fine  =  tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_fine,var_list=f_vars)

tf.summary.scalar("train_loss_coarse", train_loss_coarse, collections=['train_summary'])
tf.summary.scalar("train_loss_fine", train_loss_fine, collections=['train_summary'])

tf.summary.scalar("valid_loss_coarse", valid_loss_coarse, collections=['valid_summary'])
tf.summary.scalar("valid_loss_fine", valid_loss_fine, collections=['valid_summary'])

train_summary = tf.summary.merge_all('train_summary')
valid_summary = tf.summary.merge_all('valid_summary')

dataset,buffer_size = tf_util.prepare_2out(args.train_path,args.coarse_path,args.fine_path,
                                           batch_size,num_view=args.num_view)
dataset = dataset.repeat(args.num_repeat)
iterator = dataset.make_one_shot_iterator()
train_data, label1, label2 = iterator.get_next()

valid_dataset,valid_buffer_size = tf_util.prepare_2out(args.valid_path,args.valid_coarse_path,args.valid_fine_path,
                                                       batch_size,num_view=args.num_view)
valid_dataset = valid_dataset.repeat(args.num_repeat)
valid_iterator = valid_dataset.make_one_shot_iterator()
valid_data, valid_label1, valid_label2 = valid_iterator.get_next()
    

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(var_list=tvars, max_to_keep=1)
saver_coarse = tf.train.Saver(var_list=c_vars, max_to_keep=1)

print "Start training:"
writer = tf.summary.FileWriter(args.log_path)

if args.restore:
    print 'Restoring pretrained model...'
else:
    for n in range(args.num_repeat/2):
        # ****** Train Process ******
        loss_batch = 0.0

        for j in range(buffer_size/batch_size):
            cloud_list, coarse_list, fine_list = sess.run([train_data, label1, label2])   
            inputs, npts, gt, gt_fine  = np_util.read_dataset(cloud_list, coarse_list, fine_list)

            key_num_all_0,ball_index_0,ball_size_0,\
            key_num_all_1,ball_index_1,ball_size_1,\
            num_cloud_exp_0,num_cloud_exp_1,\
            key_index_0,key_index_1 = np_util.group(inputs, npts, args.cores, args.min_dist, args.min_points)

            feed_dict={inputs_pl:inputs,npts_pl:npts,
                       num_cloud_exp_0_pl:num_cloud_exp_0,num_cloud_exp_1_pl:num_cloud_exp_1,
                       ball_index_0_pl:ball_index_0,ball_size_0_pl:ball_size_0,
                       ball_index_1_pl:ball_index_1,ball_size_1_pl:ball_size_1,
                       gt_pl:gt,gt_fine_pl:gt_fine,train_pl:True}

            closs,closs_r,_ = sess.run([loss_coarse,loss_reg_coarse,train_coarse],feed_dict=feed_dict)

            print "Train iteration "+str(n)+"."+str(j)
            if args.print_list==True:
                print "Train list for this epoch is:"
                print cloud_list
                print coarse_list
                print fine_list

            print "coarse: " + str(closs) + ", reg: " + str(closs_r)
            loss_batch = loss_batch + closs

        print "*********"
        print "Train iterator "+ str(n) + " done"
        loss_avg = loss_batch/buffer_size*batch_size
        print "Average loss: " + str(loss_avg)
        train_summary_group = sess.run(train_summary,feed_dict={train_loss_coarse:loss_avg,train_loss_fine:0.0})
        writer.add_summary(train_summary_group, n)


        # ****** Valid Process ******
        print "------ Valid Process ------"
        loss_batch = 0.0
        
        for j in range(valid_buffer_size/batch_size):
            cloud_list, coarse_list, fine_list = sess.run([valid_data, valid_label1, valid_label2])        
            inputs, npts, gt, gt_fine  = np_util.read_dataset(cloud_list, coarse_list, fine_list)

            key_num_all_0,ball_index_0,ball_size_0,\
            key_num_all_1,ball_index_1,ball_size_1,\
            num_cloud_exp_0,num_cloud_exp_1,\
            key_index_0,key_index_1 = np_util.group(inputs, npts, args.cores, args.min_dist, args.min_points)

            feed_dict={inputs_pl:inputs,npts_pl:npts,
                       num_cloud_exp_0_pl:num_cloud_exp_0,num_cloud_exp_1_pl:num_cloud_exp_1,
                       ball_index_0_pl:ball_index_0,ball_size_0_pl:ball_size_0,
                       ball_index_1_pl:ball_index_1,ball_size_1_pl:ball_size_1,
                       gt_pl:gt,gt_fine_pl:gt_fine,train_pl:False}
            
            closs,closs_r = sess.run([loss_coarse,loss_reg_coarse],feed_dict=feed_dict)
            
            print "Valid iteration "+str(n)+"."+str(j)
            if args.print_list==True:
                print "Test list for this epoch is:"
                print cloud_list
                print coarse_list
                print fine_list

            print "coarse: " + str(closs) + ", reg: " + str(closs_r)
            loss_batch = loss_batch + closs


        print "*********"
        print "Valid iterator "+ str(n) + " done"
        loss_avg = loss_batch/valid_buffer_size*batch_size

        print "Average loss: " + str(loss_avg)
        valid_summary_group = sess.run(valid_summary,feed_dict={valid_loss_coarse:loss_avg,valid_loss_fine:0.0})       
        writer.add_summary(valid_summary_group, n)

        saver_coarse.save(sess, args.checkpoint+'coarse', global_step=n)
        print "*********"


#saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
saver_coarse.restore(sess, tf.train.latest_checkpoint(args.checkpoint))


for n in range(args.num_repeat/2,args.num_repeat):
    # ****** Train Process ******
    loss_batch = 0.0

    for j in range(buffer_size/batch_size):
        cloud_list, coarse_list, fine_list = sess.run([train_data, label1, label2])   
        inputs, npts, gt, gt_fine  = np_util.read_dataset(cloud_list, coarse_list, fine_list)    

        key_num_all_0,ball_index_0,ball_size_0,\
        key_num_all_1,ball_index_1,ball_size_1,\
        num_cloud_exp_0,num_cloud_exp_1,\
        key_index_0,key_index_1 = np_util.group(inputs, npts, args.cores, args.min_dist, args.min_points)

        feed_dict={inputs_pl:inputs,npts_pl:npts,
                   num_cloud_exp_0_pl:num_cloud_exp_0,num_cloud_exp_1_pl:num_cloud_exp_1,
                   ball_index_0_pl:ball_index_0,ball_size_0_pl:ball_size_0,
                   ball_index_1_pl:ball_index_1,ball_size_1_pl:ball_size_1,
                   gt_pl:gt,gt_fine_pl:gt_fine,train_pl:True}


        floss,floss_r,_ = sess.run([loss_fine,loss_reg_fine,train_fine],feed_dict=feed_dict)
        
        print "Train iteration "+str(n)+"."+str(j)
        if args.print_list==True:
            print "Train list for this epoch is:"
            print cloud_list
            print coarse_list
            print fine_list 

        print "fine: " + str(floss) + ", reg: " + str(floss_r)
        loss_batch = loss_batch + floss

    print "*********"
    print "Train iterator "+ str(n) + " done"
    loss_avg = loss_batch/buffer_size*batch_size
    print "Average loss: " + str(loss_avg)
    train_summary_group = sess.run(train_summary,feed_dict={train_loss_coarse:0.0,train_loss_fine:loss_avg})
    writer.add_summary(train_summary_group, n)


    # ****** Valid Process ******
    print "------ Valid Process ------"
    loss_batch = 0.0
    
    for j in range(valid_buffer_size/batch_size):
        cloud_list, coarse_list, fine_list = sess.run([valid_data, valid_label1, valid_label2])        
        inputs, npts, gt, gt_fine  = np_util.read_dataset(cloud_list, coarse_list, fine_list)        

        key_num_all_0,ball_index_0,ball_size_0,\
        key_num_all_1,ball_index_1,ball_size_1,\
        num_cloud_exp_0,num_cloud_exp_1,\
        key_index_0,key_index_1 = np_util.group(inputs, npts, args.cores, args.min_dist, args.min_points)

        feed_dict={inputs_pl:inputs,npts_pl:npts,
                   num_cloud_exp_0_pl:num_cloud_exp_0,num_cloud_exp_1_pl:num_cloud_exp_1,
                   ball_index_0_pl:ball_index_0,ball_size_0_pl:ball_size_0,
                   ball_index_1_pl:ball_index_1,ball_size_1_pl:ball_size_1,
                   gt_pl:gt,gt_fine_pl:gt_fine,train_pl:False}

        floss,floss_r = sess.run([loss_fine,loss_reg_fine],feed_dict=feed_dict)
        
        print "Valid iteration "+str(n)+"."+str(j)
        if args.print_list==True:
            print "Test list for this epoch is:"
            print cloud_list
            print coarse_list
            print fine_list

        print "fine: " + str(floss) + ", reg: " + str(floss_r)
        loss_batch = loss_batch + floss

    print "*********"
    print "Valid iterator "+ str(n) + " done"
    loss_avg = loss_batch/valid_buffer_size*batch_size
    print "Average loss: " + str(loss_avg)
    valid_summary_group = sess.run(valid_summary,feed_dict={valid_loss_coarse:0.0,valid_loss_fine:loss_avg})        
    writer.add_summary(valid_summary_group, n)

    saver.save(sess, args.save_path, global_step=n)
    print "*********"

sess.close()
