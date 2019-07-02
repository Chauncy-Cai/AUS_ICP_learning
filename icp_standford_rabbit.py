#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/3 0:52
#@Author: csc
#@File  : icp_standford_rabbit.py
'''
this file is to fully match standford rabbit (a famous data set)
'''

import open3d as o3d
import numpy as np
from icp_support import *


def rabbit_dataset():
    A1 = o3d.read_point_cloud("./bunny/data/bun000.ply")
    A2 = o3d.read_point_cloud("./bunny/data/bun045.ply")
    A3 = o3d.read_point_cloud("./bunny/data/bun090.ply")
    A4 = o3d.read_point_cloud("./bunny/data/bun180.ply")
    A5 = o3d.read_point_cloud("./bunny/data/bun270.ply")
    A6 = o3d.read_point_cloud("./bunny/data/bun315.ply")
    A7 = o3d.read_point_cloud("./bunny/data/chin.ply")
    A8 = o3d.read_point_cloud("./bunny/data/ear_back.ply")
    A9 = o3d.read_point_cloud("./bunny/data/top2.ply")
    A10 = o3d.read_point_cloud("./bunny/data/top3.ply")
    return [A1,A2,A3,A4,A5,A6,A7,A8,A9]

def icp_rabbit(datalist):
    A1,A2,A3,A4,A5,A6,A7,A8,A9 = datalist
    #o3d.draw_geometries([A1,A2,A3,A4,A5,A6,A7,A8,A9])
    train = [A1,A2,A3,A4,A5,A6,A7,A8,A9]
    train.remove(A1)
    print("start A2->A1")
    A2,R1 = icp(A2,A1)
    o3d.draw_geometries([A1, A2])
    rotate_elements(train.remove(A2),R1)
    print("start A3->A2")
    A3,R2 = icp(A3,A2)
    rotate_elements([A4,A5,A6,A7,A8,A9],R2)
    o3d.draw_geometries([A1, A2, A3])
    print("start A4->A3")
    A4,R3 = icp(A4,A3)
    rotate_elements([A5,A6],R3)
    o3d.draw_geometries([A1, A2, A3, A4])
    print("start A5->A4")
    A5,R4 = icp(A5,A4)
    rotate_elements([A6],R4)
    o3d.draw_geometries([A1, A2, A3, A4, A5])
    print("start A6->A5")
    A6,R5 = icp(A6,A5)
    o3d.draw_geometries([A1, A2, A3, A4, A5, A6])
    return

dataset = rabbit_dataset()
icp_rabbit(dataset)


