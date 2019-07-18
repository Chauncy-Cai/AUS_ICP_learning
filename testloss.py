#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/7/16 5:15
# @Author: csc
# @File  : testloss.py

import numpy as np

from loss_function import *


def test1():
    P = np.array([[1, 2, 3],
                  [6, 7, 8],
                  [9, 0, 3],
                  [0, 0, 1]])

    Q = np.array([[3, 4, 2],
                  [1, -1, -5],
                  [4, -3, -2],
                  [1, 0, 0]])

    E = np.array([[1, 3, 9, -5],
                  [0, 2, 7, -3],
                  [0, 0, 1, -1]])

    A = np.array([E, E, E])

    c = np.array([[1, 5], [3, 2], [4, 9]])

    Pc = P[[0, 1, 2]]

    for i in range(100):
        print("-----------------" + str(i) + "-----------------------")
        LOSS = lossFunction()
        LOSS.inputdata(P, Q, A, Pc, c)
        # print(LOSS.calLoss())
        R = LOSS.optimizeR()
        # print(R)
        t = LOSS.optimizeT(R)
        # print(t)
        # print(LOSS.optimize())
        LOSS.calMidloss(R)
        LOSS.calOptloss(R, t)
        LOSS.showLoss(R, t)
        P = LOSS.Pnew
        Pc = LOSS.Pcnew


def test2():
    P, Q, A, Pc, c = [1] * 5
    print(P, Q, A, Pc, c)
    for i in range(100):
        print("-----------------" + str(i) + "-----------------------")
        LOSS = lossFunction()
        if i == 0:
            LOSS.usingSample(200, 50)
            P, Q, A, Pc, c = LOSS.P, LOSS.Q, LOSS.A, LOSS.Pc, LOSS.C
        else:
            LOSS.inputdata(P, Q, A, Pc, c)
        R = LOSS.optimizeR()
        t = LOSS.optimizeT(R)
        LOSS.calMidloss(R)
        LOSS.calOptloss(R, t)
        LOSS.showLoss(R, t)
        P = LOSS.Pnew
        Pc = LOSS.Pcnew


def test3():
    P = np.array([[1, 2, 3],
                  [6, 7, 8]])

    Q = np.array([[3, 4, 2],
                  [1, -1, -5]])

    E = np.array([[1, 3, 3, 0],
                  [0, 2, 7, 0],
                  [0, 0, 1, 0]])

    A = np.array([E])

    c = np.array([[1, 5]])

    Pc = P[[0]]

    for i in range(100):
        print("-----------------" + str(i) + "-----------------------")
        LOSS = lossFunction()
        LOSS.inputdata(P, Q, A, Pc, c)
        # print(LOSS.calLoss())
        R = LOSS.optimizeR()
        # print(R)
        t = LOSS.optimizeT(R)
        # print(t)
        # print(LOSS.optimize())
        LOSS.calMidloss(R)
        LOSS.calOptloss(R, t)
        LOSS.showLoss(R, t)
        P = LOSS.Pnew
        Pc = LOSS.Pcnew


test2()
