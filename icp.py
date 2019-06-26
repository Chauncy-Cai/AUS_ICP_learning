#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/26 6:00
#@Author: csc
#@File  : icp.py

from icp_support import *

mesh1 = o3d.io.read_triangle_mesh("./OralScans/0.ply")
mesh2 = o3d.io.read_triangle_mesh("./OralScans/1.ply")
point1 = np.array(mesh1.vertices)
point2 = np.array(mesh2.vertices)
print("cal matching")
p1,p2 = pointMatching(point1,point2)
print("cal transformation")
Trans = calTransformation(p1,p2)
print("Transform...")
mesh1_ = mesh1.tranform(Trans)

mesh3 = mesh1 + mesh2
o3d.visualization.draw_geometries([mesh3],window_name="old")
mesh3 = mesh1_ + mesh2
o3d.visualization.draw_geometries([mesh3],window_name="new")