#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/27 3:26
#@Author: csc
#@File  : icp_fullmatches_test.py
'''
this file is make for valification of icp
ensure it's basic correctness
'''
from icp_support import *
import numpy as np
import open3d as o3d

A1 = o3d.geometry.PointCloud()
points = generate_points(message=1,d=1,count=500)
A1.points = o3d.Vector3dVector(points)
A2 = o3d.geometry.PointCloud()
points = generate_points(message=1,d=1,count=1000)
A2.points = o3d.Vector3dVector(points)
Tran = vec2pose(np.random.randn(6)*0.1)
print(Tran)
#Tran = transform(0.2, 0.3, 0.4, 1, 2, 3).T
A1.transform(Tran)
o3d.estimate_normals(A2)
A2.normalize_normals()
p2norm = np.array(A2.normals)
# o3d.draw_geometries([A1 + A2], window_name="epo ",width=300,height=300)

i = 0

def custom_draw_geometry_with_view_tracking(meshes):
    def track_view(vis):
        global i, A1, A2
        for i in range(30):
            p1 = np.array(A1.points)
            p2 = np.array(A2.points)
            p1, p2, indice = point_matching(p1, p2)
            print("[" + str(i) + "/91] loss:" + str(icploss(p1, p2)))
            weight = np.linalg.norm((p1-p2),axis=1)
            weight = (1e-1/(weight+1e-1))[:,np.newaxis]
            Tran0 = cal_transformation(p1, p2,weight)          #如果不需要ilrs不要输入      #p2p
            #p2norm_use = p2norm [indice]                      #p2pl
            #Tran0 = cal_transformation_p2pl(p1, p2,p2norm_use) #p2pl
            A1.transform(Tran0)
            i += 1
            vis.update_geometry()
    o3d.draw_geometries_with_animation_callback(meshes, track_view)

custom_draw_geometry_with_view_tracking([A1, A2])
