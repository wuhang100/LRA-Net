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
import pcl
#import open3d as o3d
sys.path.append("./utils")
import tf_util
import np_util
sys.path.append("./models")
import model
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-model_type', default='fine')
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-num_repeat', type=int, default=1)

parser.add_argument('-num_view', type=int, default=30)
parser.add_argument('-test_path', default='../data/cartest/partial')
parser.add_argument('-coarse_path', default='../data/cartest/full_1k')
parser.add_argument('-fine_path', default='../data/cartest/full_1w')

parser.add_argument('-fine_model_path', default='./restore/fine/')
parser.add_argument('-pretrain_model_path', default='./restore/pretrain/')
parser.add_argument('-pcd_path_coarse', default='./output/pcd_file/coarse/')
parser.add_argument('-pcd_path_fine', default='./output/pcd_file/fine/')
parser.add_argument('-csv_path', default='./output/csv/')
parser.add_argument('-test_version', type=int, default=0)

parser.add_argument('-cores', type=int, default=[64,16])
parser.add_argument('-min_dist', type=float, default=[0.25,0.5])
parser.add_argument('-min_points', type=int, default=[8,0])
parser.add_argument('-nheads', type=int, default=16)

parser.add_argument('-bn', type=bool, default=False)
parser.add_argument('-in_drop', type=float, default=0.0)
parser.add_argument('-coef_drop', type=float, default=0.0)
parser.add_argument('-reg_coef', type=float, default=0.1)

parser.add_argument('-print_list', action='store_true', default=False)

args = parser.parse_args()

"""
Model parameters
"""
batch_size = 1
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


def get_loss(coarse, fine, gt, gt_fine, reg_coef, is_training):
    
    reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_coarse = [reg_c for reg_c in reg if not 'decoder_fine' in reg_c.name]
    reg_fine = [reg_f for reg_f in reg if 'decoder_fine' in reg_f.name]

    loss_reg_coarse = reg_coef * tf.cast(is_training,tf.float32)*tf.reduce_mean(reg_coarse)
    loss_reg_fine = 0.2 * reg_coef * tf.cast(is_training,tf.float32)*tf.reduce_mean(reg_fine)

    loss_coarse = tf_util.earth_mover(coarse,gt) + loss_reg_coarse
    loss_fine = tf_util.chamfer(fine, gt_fine) + loss_reg_fine

    return loss_coarse, loss_reg_coarse, loss_fine, loss_reg_fine

MODEL = model.LAN(inputs_pl,npts_pl,args.cores,
                  num_cloud_exp_0_pl,num_cloud_exp_1_pl,
                  ball_index_0_pl, ball_size_0_pl,ball_index_1_pl, ball_size_1_pl,
                  args.nheads, False, args.in_drop, args.coef_drop, args.bn, False)

pc_coarse = MODEL.coarse
pc_fine = MODEL.fine
loss_coarse,loss_reg_coarse,loss_fine,loss_reg_fine = get_loss(pc_coarse, pc_fine, gt_pl, gt_fine_pl, args.reg_coef, False)

tvars = tf.trainable_variables()
tf_util.printvar (tvars)
c_vars = [var for var in tvars if not 'decoder_fine' in var.name]
f_vars = [var for var in tvars if 'decoder_fine' in var.name]


dataset,buffer_size = tf_util.prepare_2out(args.test_path,args.coarse_path,args.fine_path,
                                           batch_size,num_view=args.num_view,shuffle=False)
dataset = dataset.repeat(args.num_repeat)
iterator = dataset.make_one_shot_iterator()
test_data, label1, label2 = iterator.get_next()
    

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(var_list=tvars, max_to_keep=1)
saver_coarse = tf.train.Saver(var_list=c_vars, max_to_keep=1)


if args.model_type=='coarse':

    f = io.open(args.csv_path+str(args.test_version)+'_testresult_growpc_coarse.csv','wb')
    csv_writer = csv.writer(f)
    
    print 'Restoring coarse model...'
    saver_coarse.restore(sess, tf.train.latest_checkpoint(args.pretrain_model_path))
    
    print "------ Test Process ------"    
    for j in range(buffer_size):
        
        cloud_list, coarse_list, fine_list = sess.run([test_data, label1, label2])        
        inputs, npts, gt, gt_fine  = np_util.read_dataset(cloud_list, coarse_list, fine_list)

        key_num_all_0,ball_index_0,ball_size_0,\
        key_num_all_1,ball_index_1,ball_size_1,\
        num_cloud_exp_0,num_cloud_exp_1,\
        key_index_0,key_index_1 = np_util.group(inputs, npts, args.cores, args.min_dist, args.min_points)

        feed_dict={inputs_pl:inputs,npts_pl:npts,
                   num_cloud_exp_0_pl:num_cloud_exp_0,num_cloud_exp_1_pl:num_cloud_exp_1,
                   ball_index_0_pl:ball_index_0,ball_size_0_pl:ball_size_0,
                   ball_index_1_pl:ball_index_1,ball_size_1_pl:ball_size_1,
                   gt_pl:gt,gt_fine_pl:gt_fine}
        
        cout,closs,closs_r = sess.run([pc_coarse,loss_coarse,loss_reg_coarse],feed_dict=feed_dict)
		pcd_name = cloud_list[0]
		pcd_name = pcd_name.split('/')
		pcd_name = pcd_name[-1]

        if args.print_list==True:
            print "Test for: " + pcd_name
            print "Coarse and fine file: "+coarse_list[0] + ", " + fine_list[0]

        print "Coarse loss is "+ str(closs) + ", reg: " + str(closs_r)

        pcd_out = pcl.PointCloud(1024)
        pcd_out.from_array(cout[0])

        save_name = args.pcd_path_coarse + str(args.test_version) + '_' + pcd_name
		#print save_name
	
        pcl.save(pcd_out, save_name)


        csv_writer.writerow([pcd_name,str(closs),str(closs_r)])

    f.close()


if args.model_type=='fine':

    f = io.open(args.csv_path+str(args.test_version)+'_testresult_growpc_fine.csv','wb')
    csv_writer = csv.writer(f)
    
    print 'Restoring fine model...'
    saver.restore(sess, tf.train.latest_checkpoint(args.fine_model_path))
    
    print "------ Test Process ------"    
    for j in range(buffer_size):
        
        cloud_list, coarse_list, fine_list = sess.run([test_data, label1, label2])        
        inputs, npts, gt, gt_fine  = np_util.read_dataset(cloud_list, coarse_list, fine_list)

        key_num_all_0,ball_index_0,ball_size_0,\
        key_num_all_1,ball_index_1,ball_size_1,\
        num_cloud_exp_0,num_cloud_exp_1,\
        key_index_0,key_index_1 = np_util.group(inputs, npts, args.cores, args.min_dist, args.min_points)

        feed_dict={inputs_pl:inputs,npts_pl:npts,
                   num_cloud_exp_0_pl:num_cloud_exp_0,num_cloud_exp_1_pl:num_cloud_exp_1,
                   ball_index_0_pl:ball_index_0,ball_size_0_pl:ball_size_0,
                   ball_index_1_pl:ball_index_1,ball_size_1_pl:ball_size_1,
                   gt_pl:gt,gt_fine_pl:gt_fine}
        
        fout,floss,floss_r = sess.run([pc_fine,loss_fine,loss_reg_fine],feed_dict=feed_dict)
		
		pcd_name = cloud_list[0]
		pcd_name = pcd_name.split('/')
		pcd_name = pcd_name[-1]

        if args.print_list==True:
            print "Test for: " + pcd_name
            print "Coarse and fine file: "+coarse_list[0] + ", " + fine_list[0]

        print "Fine loss is "+ str(floss) + ", reg: " + str(floss_r)

        pcd_out = pcl.PointCloud(16384)
        pcd_out.from_array(fout[0])

        save_name = args.pcd_path_fine + str(args.test_version) + '_' + pcd_name
        pcl.save(pcd_out, save_name)

        csv_writer.writerow([pcd_name,str(floss),str(floss_r)])

    f.close() 

sess.close()
