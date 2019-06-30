#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/26 6:00
#@Author: csc
#@File  : icp.py
'''
this file is aim at applicatoin of icp
'''


from icp_support import *
import open3d as o3d

A1 = o3d.read_point_cloud("./OralScans/0.ply")
A2 = o3d.read_point_cloud("./OralScans/1.ply")
#o3d.visualization.draw_geometries([pcd1+pcd2],window_name="old",width=300,height=300)
o3d.estimate_normals(A2)
p2norm = np.array(A2.normals)
i = 0

def custom_draw_geometry_with_view_tracking(meshes):
    def track_view(vis):
        global i, A1, A2
        if i%10==0:
            point1 = np.array(A1.points)
            point2 = np.array(A2.points)
            p1, p2, indice = pointMatching(point1,point2)
            weight = np.linalg.norm((p1 - p2), axis=1)
            avg_weight = 0.45 #np.average(weight)
            #print(avg_weight)
            weight = (avg_weight / (weight + avg_weight))[:, np.newaxis]
            print("loss" + str(i) + ":" + str(icploss(p1, p2, weight)))
            #Trans = cal_transformation(p1, p2, weight)                   #p2p
            p2norm_use = p2norm[indice]                         #p2pl
            Trans = cal_transformation_p2pl(p1, p2, p2norm_use)   #p2pl
            A1.transform(Trans)
            vis.update_geometry()
        i += 1
    o3d.draw_geometries_with_animation_callback(meshes, track_view)

custom_draw_geometry_with_view_tracking([A1, A2])
