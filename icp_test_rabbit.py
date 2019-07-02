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
    return [A1,A2,A3,A4,A5,A6,A7,A8,A9,A10]

def icp_rabbit(datalist):
    A1,A2,A3,A4,A5,A6,A7,A8,A9,A10 = datalist
    o3d.draw_geometries([A8,A9])
    print("start A3->A2")
    A8,R9 = icp(A8,A9)
    o3d.draw_geometries([A8,A9])
    return

dataset = rabbit_dataset()
icp_rabbit(dataset)


