#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/26 6:00
#@Author: csc
#@File  : icp.py

import open3d as o3d
import numpy as np

def pointMatching(pointlist1,pointlist2):
    matchingList = []
    for i in range(pointlist1):
        dist = None
        match = None
        for j in range(pointlist2):
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
    return matchingList



print("Test IO for mesh ...")
mesh0 = o3d.io.read_triangle_mesh("./OralScans/0.ply")
mesh1 = o3d.io.read_triangle_mesh("./OralScans/1.ply")
point0 = np.array(mesh0.vertices)
point1 = np.array(mesh1.vertices)
matchingList = pointMatching(point0,point1)

mesh3 = o3d.geometry.TriangleMesh(o3d.utility.DoubleVector(point1))
mesh3 = mesh0 + mesh1
o3d.visualization.draw_geometries([mesh3])
#o3d.io.write_triangle_mesh("copy_of_1.ply",mesh)