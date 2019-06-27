#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/27 3:26
#@Author: csc
#@File  : icp_fullmatches_test.py

from icp_support import *
from TransformGeneration import *
import numpy as np
import open3d as o3d

A1 = o3d.geometry.PointCloud()
points = generate_points(message=1,d=1,count=500)
A1.points = o3d.utility.Vector3dVector(points)
A2 = o3d.geometry.PointCloud()
points = generate_points(message=1,d=1)
A2.points = o3d.utility.Vector3dVector(points)
Tran = transform(0.2, 0.3, 0.4, 1, 2, 3).T
A1.transform(Tran)
o3d.visualization.draw_geometries([A1 + A2], window_name="epo ",width=300,height=300)
showlist = [1,11,21,31]
for i in range(32):
    mark = 0
    p1 = np.array(A1.points)
    p2 = np.array(A2.points)
    p1, p2 = pointMatching(p1, p2)
    print("[" + str(i) + "/91] loss:" + str(icploss(p1, p2)))
    Tran0 = calTransformation(p1, p2).T
    A1.transform(Tran0)
    if (i in showlist)|(mark==1):
        o3d.visualization.draw_geometries([A1 + A2], window_name="epo "+str(i),width=300,height=300)

