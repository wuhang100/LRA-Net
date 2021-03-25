# -*- coding: utf-8 -*-

import sys
import numpy as np
import heapq
import pcl

def ball_cluster(cloud, cores, distance, min_points=32):
    """
    Inputs: cloud [pts,3+...]
    """
    num_pts = np.shape(cloud)[0]
    key_index_all = []
    key_distance = 10.0*np.ones([num_pts],dtype=np.float32)
    
    ball_index_all = []
    ball_size_all = []
    
    key_index = np.random.randint(0, num_pts)
    
    i = 0
    while i < cores:
        dist = np.sqrt(np.sum((cloud[:,0:3]-cloud[key_index,0:3])**2,-1))
        ball_index = np.where(dist<distance)[0]
        ballsize = np.shape(ball_index)[0]
        
        if ballsize >= min_points:
            key_index_all.append(key_index)
            ball_index_all.append(ball_index)
            ball_size_all.append(ballsize)
            i += 1
 
        mask = (dist < key_distance).astype(float)
        key_distance = key_distance - np.multiply(mask,key_distance) + np.multiply(mask,dist)
        key_index = np.argmax(key_distance)
           
    key_index_all = np.hstack(key_index_all)
    ball_index_all = np.hstack(ball_index_all)
    ball_size_all = np.hstack(ball_size_all)
    return key_index_all, ball_index_all, ball_size_all


def batch_ball_cluster(cloud_full, npts_full, cores, distance, min_points=32):
    """
    Inputs: cloud [1,None,3]
    """
    cloud_full = np.squeeze(cloud_full)
    batch_size = np.shape(npts_full)[0]
    key_index_all = []
    ball_index_all = []
    ball_size_all = []
    key_num_all = []
    for i in range(0,batch_size):
        cloud = cloud_full[np.sum(npts_full[:i]):np.sum(npts_full[:i+1]),0:3]
        key_index, ball_index, ball_size = ball_cluster(cloud, cores, distance, min_points)
        key_index = key_index + np.sum(npts_full[:i])
        ball_index = ball_index + np.sum(npts_full[:i])
        key_index_all.append(key_index)
        ball_index_all.append(ball_index)
        ball_size_all.append(ball_size)
        key_num_all.append(np.shape(key_index)[0])
        
    
    key_index_all = np.hstack(key_index_all)
    ball_index_all = np.hstack(ball_index_all)
    ball_size_all = np.hstack(ball_size_all)
    key_num_all = np.hstack(key_num_all)
    
    return key_num_all,key_index_all,ball_index_all,ball_size_all

def get_expand_num(batch_size,key_num_all,ball_size):
    ball_num_all = []
    ball_size_start = 0
    for i in range(batch_size):
        ball_size_end = ball_size_start+key_num_all[i]
        ball_num_all.append(np.sum( ball_size[ball_size_start:ball_size_end] ))
        ball_size_start = ball_size_end
    ball_num_all = np.hstack(ball_num_all)
    return ball_num_all

def group(inputs, npts, cores, min_dist, min_points):
    key_num_all_0,key_index_0,ball_index_0,ball_size_0 = batch_ball_cluster(inputs, npts, cores[0], min_dist[0], min_points[0])
    key_points_0 = np.expand_dims(inputs[0,key_index_0,0:3],axis=0)   
    key_num_all_1,key_index_1,ball_index_1,ball_size_1 = batch_ball_cluster(key_points_0, key_num_all_0, cores[1], min_dist[1],  min_points[1])
    num_cloud_exp_0 = get_expand_num(np.shape(npts)[0],key_num_all_0,ball_size_0)
    num_cloud_exp_1 = get_expand_num(np.shape(npts)[0],key_num_all_1,ball_size_1)    
    return key_num_all_0,ball_index_0,ball_size_0,key_num_all_1,ball_index_1,ball_size_1,num_cloud_exp_0,num_cloud_exp_1,key_index_0,key_index_1

def read_dataset(cloud_list, coarse_list, fine_list):
    npts = []
    pcd = []
    for p in cloud_list:
        pc_batch = np.asarray(pcl.load(p))            
        inpts = np.shape(pc_batch)[0]
        npts.append(inpts)
        pcd.append(pc_batch)
    npts = np.array(npts)
    pcd = np.vstack(pcd)
    pcd = np.expand_dims(pcd, axis=0)
    
    gt_pc = []
    for p in coarse_list:
        gt_batch = np.asarray(pcl.load(p))            
        gt_pc.append(gt_batch)
    gt_pc = np.reshape(gt_pc,[-1,1024,3])

    gt_pc_fine = []
    for pf in fine_list:
        gt_batch_fine = np.asarray(pcl.load(pf))            
        gt_pc_fine.append(gt_batch_fine)
    gt_pc_fine = np.reshape(gt_pc_fine,[-1,16384,3])

    return pcd, npts, gt_pc, gt_pc_fine



  
