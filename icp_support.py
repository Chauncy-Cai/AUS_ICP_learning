#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/27 3:24
#@Author: csc
#@File  : icp_support.py

import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
from math import exp

def check(P1,P2):
    p = True
    for i in range(len(P1)):
        for j in range(len(P1[1])):
            if P1[i][j]!=P2[i][j]:
                print(False)
                return False
    print(True)
    return True

def pointMatching(pointlist1,pointlist2):
    pointlist1 = np.array(pointlist1)
    pointlist2 = np.array(pointlist2)
    nbrs = NearestNeighbors(n_neighbors=1).fit(pointlist2)
    _, indices = nbrs.kneighbors(pointlist1)
    p1 = np.array([pointlist1[i] for i in range(len(pointlist1))])
    p2 = np.array([pointlist2[indices[i][0]] for i in range(len(pointlist1))])
    return p1, p2

def calTransformation(p1,p2):
    #1=>2 p1->q2
    def calDense(points):
        return np.sum(points,axis=0)/len(points)
    def calverse(dense,points):
        return np.array([ k-dense for k in points])
    def calW(p1,p2):
        sum = np.zeros((3,3))
        for i in range(len(p1)):
            sum += np.dot(p2.T, p1)#p2,p1本来就是一维的，转智也没有什么变化
        return sum
    def Rt2mat(R,t):
        Re = np.append(R, t, axis=1)
        return np.append(Re,np.array([[0, 0, 0, 1]]), axis=0)
    d1 = calDense(p1)
    d2 = calDense(p2)
    p1 = calverse(d1, p1)
    p2 = calverse(d2, p2)
    W = calW(p1, p2)
    u, _, vt = np.linalg.svd(W)
    R = np.dot(u, vt)
    d1 = np.array([d1]).T
    d2 = np.array([d2]).T
    t = d2 - np.dot(R, d1)
    return Rt2mat(R, t)

def icploss(p1,p2):
    return np.linalg.norm(p1-p2)/len(p1)

def generate_points(d=100,count=5000,message=0,demo=0):
    if demo==1:
        x=[[1,0,0],
           [2,0,0],
           [3,0,0]]
        return np.array(x)
    if message==1:
        print("generation points")
    dataset = []
    for i in range(count):
        x = random.uniform(-5,5)
        y = random.uniform(-5,5)
        g = exp(-(x**2+y**2)/2/d/d)/(2*3.14*d*d)*25
        dataset.append([x,y,g])
    dataset = np.array(dataset)
    return dataset