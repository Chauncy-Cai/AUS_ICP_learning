#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/6/27 3:24
# @Author: csc
# @File  : icp_support.py

####
# point2plane：缺少法向量
# irls：缺少优化过程
####

import random
from math import exp

import numpy as np
from sklearn.neighbors import NearestNeighbors


def check(P1, P2):
    p = True
    for i in range(len(P1)):
        for j in range(len(P1[1])):
            if P1[i][j] != P2[i][j]:
                print(False)
                return False
    print(True)
    return True


def pointMatching(pointlist1, pointlist2):
    pointlist1 = np.array(pointlist1)
    pointlist2 = np.array(pointlist2)
    nbrs = NearestNeighbors(n_neighbors=1).fit(pointlist2)
    _, indices = nbrs.kneighbors(pointlist1)
    p1 = np.array([pointlist1[i] for i in range(len(pointlist1))])
    p2 = np.array([pointlist2[indices[i][0]] for i in range(len(pointlist1))])
    indices = np.array(indices).T[0]
    return p1, p2, indices


def transform(alpha, beta, gamma, x, y, z):
    mov = np.array([[x, y, z]]).T
    A = [[1, 0, 0],
         [0, np.cos(alpha), -np.sin(alpha)],
         [0, np.sin(alpha), np.cos(alpha)]]
    B = [[np.cos(beta), 0, np.sin(beta)],
         [0, 1, 0],
         [-np.sin(beta), 0, np.cos(beta)]]
    C = [[np.cos(gamma), -np.sin(gamma), 0],
         [np.sin(gamma), np.cos(gamma), 0],
         [0, 0, 1]]
    W = np.dot(A, B)
    W = np.dot(W, C)
    # rot = W
    Tran = np.append(W, mov, axis=1)
    Tran = np.append(Tran, np.array([[0, 0, 0, 1]]), axis=0)
    return Tran


def calTransformation(p1, p2):
    # 1=>2 p1->q2
    def calDense(points):
        return np.sum(points, axis=0) / len(points)

    def calverse(dense, points):
        return np.array([k - dense for k in points])

    def calW(p1, p2):
        sum = np.zeros((3, 3))
        for i in range(len(p1)):
            sum += np.dot(p2.T, p1)  # p2,p1本来就是一维的，转智也没有什么变化
        return sum

    def Rt2mat(R, t):
        Re = np.append(R, t, axis=1)
        return np.append(Re, np.array([[0, 0, 0, 1]]), axis=0)

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


def icploss(p1, p2):
    return np.linalg.norm(p1 - p2) / len(p1)


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


def calTransformation_p2pl(p1, p2, normofp2):  # p1->p2
    cross = [np.cross(p1[i], normofp2[i]) for i in range(len(p1))]
    cross = np.array(cross)
    print("cross",np.shape(cross))
    print("normofp2", np.shape(normofp2))
    Para = np.append(cross, normofp2, axis=1)  # n*6
    B = np.sum(((p1 - p2) * normofp2), axis=1)
    B = np.dot(np.array([B]), Para).T
    A = np.dot(Para.T, Para)
    X = np.linalg.solve(A, -B).T[0]
    Trans = transform(X[0], X[1], X[2], X[3], X[4], X[5])
    return Trans


'''
def calTransformation_irls(p1,p2): #p1->p2
    #|x-Ay|
    A = formA(p1, norm)
    b = formb(p1, p2, norm)
    u,d,vt = np.linalg.svd(A)
    for i in range(len(d)):
        if(d[i] == 0):
            continue
        d[i] = 1./d[i] #verse
    D = np.eyes(len(p1),6) #n*6
    for i in range(len(d)):
        D[i][i] = d[i]
    A_star = np.dot(n,D)
    A_star = np.dot(A_star,vt)#A = U*D_star*VT
    x = np.dot(A_star,b)
    Trans = transform(x[0], x[1], x[2], x[3], x[4], x[5])
    return Trans

'''
