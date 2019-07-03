#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/6/27 3:24
# @Author: csc
# @File  : icp_support.py


import random
import numpy as np
import open3d as o3d
from math import exp
from sklearn.neighbors import NearestNeighbors

'''
not be used always to be the temp function
'''
def check(P1, P2):
    p = True
    for i in range(len(P1)):
        for j in range(len(P1[1])):
            if P1[i][j] != P2[i][j]:
                print(False)
                return False
    print(True)
    return True

'''
INPUT:
pointlist1 point cloud1
pointlist2 point cloud2
OUTPUT:
p1:points of point cloud1
p2:points matching with p1 respectively
indices: index of pointlist2, matching best of p1
'''


def point_matching(pointlist1, pointlist2, color1=None, color2=None, hascolor=0):
    pointlist1 = np.array(pointlist1)
    pointlist2 = np.array(pointlist2)
    if hascolor == 0:
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=10).fit(pointlist2)
        _, indices = nbrs.kneighbors(pointlist1)
    else:
        point_and_color1 = np.append(pointlist1, color1, axis=1)
        point_and_color2 = np.append(pointlist2, color2, axis=1)
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=10).fit(point_and_color2)
        _, indices = nbrs.kneighbors(point_and_color1)
    p1 = np.array([pointlist1[i] for i in range(len(pointlist1))])
    p2 = np.array([pointlist2[indices[i][0]] for i in range(len(pointlist1))])
    indices = np.array(indices).T[0]
    return p1, p2, indices

'''
cross operation
INPUT:
r: rx?
OUTPUT:
R: rx? show in a matrix way
'''
def cross_op(r):
    R = np.zeros((3, 3))
    R[0, 1] = -r[2]
    R[0, 2] = r[1]
    R[1, 2] = -r[0]
    R = R - R.T
    return R

'''
translation and rotation vec -> matrix T
'''
def vec2pose(translation_rotation_vec):
    t = translation_rotation_vec[:3]
    r = translation_rotation_vec[3:]
    theta = np.linalg.norm(r, 2)
    k = r / theta
    """ Roduiguez"""
    R = np.cos(theta)*np.eye(3)+np.sin(theta)*cross_op(k)+(1-np.cos(theta))*np.outer(k, k)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
'''
pack rotation matrix and translation vector into a transposition matrix
'''
def pack(R, t):
    t = np.squeeze(t)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

'''
calculation the best transposition matrix from p1->p2
INPUT
p1:point cloud 1
p2:point cloud 2
weight: when using irls, weight become the weight influenced the result
OUTPUT : transposition
'''
def cal_transformation(p1, p2,weight="none"):
    # 1=>2 p1->q2
    d1 = np.mean(p1, axis=0)
    d2 = np.mean(p2, axis=0)
    p1 = p1 - d1
    p2 = p2 - d2
    if type(weight)!=str:
        #print(weight,p2)
        p2 = weight*p2 #引入权重矩阵
        #print(p2)
    W = p2.T.dot(p1)
    u, _, vt = np.linalg.svd(W)
    R = np.dot(u, vt)
    d1 = np.array([d1]).T
    d2 = np.array([d2]).T
    t = d2 - np.dot(R, d1)
    return pack(R, t)

'''
calculate the loss of algorithm of icp.
'''
def icploss(p1, p2, weight="none",point2plane=0,normofp2 = None):
    if point2plane==0:
        if type(weight)!=str:
            return np.linalg.norm(weight * (p1-p2))/len(p1)
        else:
            return np.linalg.norm(p1 - p2) / len(p1)
    else: #point 2 plane
        L = (p1 - p2) * normofp2
        if type(weight)!=str:
            return (weight*L).dot(L.T)/len(p1)
        else:
            return L.dot(L.T)/len(p1)
'''
To generate test point cloud
'''
def generate_points(d=100, count=5000, message=0, demo=0):
    if demo == 1:
        x = [[1, 0, 0],
             [2, 0, 0],
             [3, 0, 0]]
        return np.array(x)
    if message == 1:
        print("generation points")
    dataset = []
    for i in range(count):
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        g = exp(-(x ** 2 + y ** 2) / 2 / d / d) / (2 * 3.14 * d * d) * 25
        dataset.append([x, y, g])
    dataset = np.array(dataset)
    return dataset


"""
    point2plane
    p1: point cloud 1
    p2: point cloud 2
    normofp2: normal of point cloud 2
"""
def cal_transformation_p2pl(p1, p2, normofp2,weight = "none"):  # p1->p2
    cross = np.cross(p1, normofp2)
    Para = np.append(normofp2, cross, axis=1)  # n*6
    b = np.sum(((p1 - p2) * normofp2), axis=1) # n*1
    #print(b.shape)
    if type(weight)!=str:
        #print(b.shape,weight.shape)
        b = weight * b
        Para = weight * Para
    b = np.dot(np.array([b]), Para).T
    A = np.dot(Para.T, Para)
    delta_translation_rotation = np.linalg.solve(A, -b).T[0]
    T = vec2pose(delta_translation_rotation)

    return T

'''
the icp function, make it good to be called
INPUT:
A1,A2 two type of point cloud
OUTPUT:
A1,A2 When converge
'''
def icp(A1,A2,weight=0,norm = 0,detail=0):
    # put A1->A2
    loss = 1000
    o3d.estimate_normals(A2)
    norm0 = np.array(A2.normals)
    Trans0 = np.eye(4)
    i = 0
    while(1):
        i+=1
        point1 = np.array(A1.points)
        point2 = np.array(A2.points)
        p1,p2,indice = point_matching(point1,point2)
        norm = norm0[indice]
        weight = np.linalg.norm((p1 - p2), axis=1)
        avg_weight = 0.45  # np.average(weight)
        weight = (avg_weight / (weight + avg_weight))[:, np.newaxis]
        newloss = icploss(p1, p2, weight)
        if ((newloss) > 1e-5)&(i<80):
            loss = newloss
            if (i%10==0)|(detail==1):
                print("converging iter " + str(i) + ", now loss is "+ str(newloss))
        else:
            print("converged: loss is "+str(newloss))
            break
        Trans = cal_transformation(p1, p2, weight)
        #Trans = cal_transformation_p2pl(p1,p2,normofp2=norm,weight=weight)
        A1.transform(Trans)
        Trans0 = Trans0.dot(Trans)
    print(Trans0)
    return A1,Trans0

def rotate_elements(pcd_list,R):
    for A in pcd_list:
        A.transform(R)
    return
