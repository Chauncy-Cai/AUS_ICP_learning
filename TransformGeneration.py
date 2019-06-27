#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/27 3:30
#@Author: csc
#@File  : TransformGeneration.py

import numpy as np

def transform(alpha,beta,gamma,x,y,z):
    mov = np.array([[x, y, z]]).T
    A = [[1,0,0],
        [0,np.cos(alpha), -np.sin(alpha)],
        [0,np.sin(alpha), np.cos(alpha)]]
    B = [[np.cos(beta), 0, np.sin(beta)],
         [0, 1, 0],
         [-np.sin(beta), 0, np.cos(beta)]]
    C = [[np.cos(gamma), -np.sin(gamma), 0],
         [np.sin(gamma), np.cos(gamma), 0],
         [0, 0, 1]]
    W = np.dot(A,B)
    W = np.dot(W,C)
    #rot = W
    Tran = np.append(W, mov, axis=1)
    Tran = np.append(Tran, np.array([[0, 0, 0, 1]]), axis=0)
    return Tran