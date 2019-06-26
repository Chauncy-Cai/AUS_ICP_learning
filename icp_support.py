#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/27 3:24
#@Author: csc
#@File  : icp_support.py

import open3d as o3d
import numpy as np

def pointMatching(pointlist1,pointlist2):
    matchingList = []
    print((len(pointlist1),len(pointlist2)))
    p1, p2 = [], []
    for i in range(len(pointlist1)):
    #for i in range(5000):
        if i % 100 == 0:
            print("[" + str(i) + "/19600]")
        dist = None
        match = None
        for j in range(len(pointlist2)):
            temp = np.linalg.norm(pointlist1[i]-pointlist2[j])
            if temp == 0:
                match = [i,j]
                break
            if dist is None:
                dist = temp
                match = [i,j]
                continue
            if dist > temp:
                dist = temp
                match = [i,j]
        matchingList.append(match)
    p1 = np.array([pointlist1[match[0]] for match in matchingList])
    p2 = np.array([pointlist2[match[1]] for match in matchingList])
    return p1,p2

def calTransformation(p1,p2):
    #1=>2 p1->q2
    def calDense(points):
        return np.sum(points,axis=1)/len(points)
    def calverse(dense,points):
        return np.array([ k-dense for k in points])
    def calW(p1,p2):
        sum = np.zeros((3,3))
        for i in range(len(p1)):
            sum += np.dot(p1[i].T,p2[i])
        return sum
    def Rt2mat(R,t):
        Re = np.append(R,t,axis = 1)
        return np.append(Re,np.array([0,0,0,1]))
    d1 = calDense(p1)
    d2 = calDense(p2)
    p1 = calverse(d1,p1)
    p2 = calverse(d2,p2)
    W = calW(p1,p2)
    u,_,vt = np.linalg.svd(W)
    R = np.dot(u,vt)
    t = d2 - np.dot(R,d1)
    return Rt2mat