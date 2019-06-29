#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/27 3:26
#@Author: csc
#@File  : icp_fullmatches_test.py

from icp_support import *
import numpy as np
import open3d as o3d

A1 = o3d.PointCloud()
points = generate_points(message=1,d=1,count=500)
A1.points = o3d.Vector3dVector(points)
A2 = o3d.PointCloud()
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

        if i % 100 == 0:
            mark = 0
            p1 = np.array(A1.points)
            p2 = np.array(A2.points)
            p1, p2, indice = pointMatching(p1, p2)
            print("[" + str(i) + "/91] loss:" + str(icploss(p1, p2)))
            Tran0 = calTransformation(p1, p2)                #p2p
            #p2norm_use = p2norm [indice]                      #p2pl
            #Tran0 = cal_transformation_p2pl(p1, p2,p2norm_use) #p2pl
            A1.transform(Tran0)
            #f (i in showlist)|(mark==1):
            #   o3d.draw_geometries([A1 + A2], window_name="epo "+str(i),width=1920,height=1024)
            vis.update_geometry()

        i += 1

    o3d.draw_geometries_with_animation_callback(meshes, track_view)

custom_draw_geometry_with_view_tracking([A1, A2])
