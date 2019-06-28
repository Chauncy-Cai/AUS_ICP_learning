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
    return p1, p2


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


def calTransformation_p2pl(p1, p2):  # p1->p2
    def calnorm(p1):  # 利用叉乘
        # simple_vesion
        nbrs = NearestNeighbors(n_neighbors=2).fit(p1)
        _, indices = nbrs.kneighbors(p1)
        norms = []
        for i in range(len(indices)):
            index1, index2 = indices[i][0], indices[i][1]
            vec = np.cross(p1[index1], p1[index2])
            vec = vec / np.linalg.norm(vec)  # normalize
            norms.append(vec)
        return np.array(norms)
        '''
        nbrs = NearestNeighbors(n_neighbors=50).fit(p1)
        _,indices = nbrs.kneighbors(p1)
        N = p1[indices]
        norms = []
        for i in range(len(N)):
            temp = o3d.geometry.PointCloud()
            temp.points = o3d.Vector3dVector(np.array(N[i]))
        '''

    def formb(source, dest, norm):
        s_by_n = source * norm
        d_by_n = dest * norm
        return np.sum(d_by_n, axis=1) - np.sum(s_by_n, axis=1)

    def formA(source, norm):
        A = []
        for i in range(len(norm)):
            normi = norm[i]
            sourcei = source[i]
            a1 = normi[2] * sourcei[1] - normi[1] * sourcei[2]
            a2 = normi[0] * sourcei[2] - normi[2] * sourcei[0]
            a3 = normi[1] * sourcei[0] - normi[0] * sourcei[1]
            A.append([a1, a2, a3, normi[0], normi[1], normi[2]])
        return np.array(A)
    #too large
    ranlist = [random.randint(1, len(p1) - 1) for i in range(50)]
    p1 = p1[ranlist]
    p2 = p2[ranlist]
    ############
    norm = calnorm(p1)
    A = formA(p1, norm)
    b = formb(p1, p2, norm)
    u, d, vt = np.linalg.svd(A)
    for i in range(len(d)):
        if (d[i] == 0):
            continue
        d[i] = 1. / d[i]  # verse
    D = np.eye(len(p1), 6)  # n*6
    for i in range(len(d)):
        D[i][i] = d[i]
        A_star = np.dot(vt.T, D.T)
    A_star = np.dot(A_star, u)  # A = U*D_star*VT
    x = np.dot(A_star, b)
    Trans = transform(x[0], x[1], x[2], x[3], x[4], x[5])
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
