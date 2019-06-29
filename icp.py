#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/26 6:00
#@Author: csc
#@File  : icp.py

from icp_support import *
import open3d as o3d

A1 = o3d.read_point_cloud("./OralScans/0.ply")
A2 = o3d.read_point_cloud("./OralScans/1.ply")
#o3d.visualization.draw_geometries([pcd1+pcd2],window_name="old",width=300,height=300)
#A1 = o3d.PointCloud()
#A2 = o3d.PointCloud()
#A1.points = pcd1.points
#A2.points = pcd2.points
o3d.estimate_normals(A2)
p2norm = np.array(A2.normals)

#showlist = [1,20, 50, 90]
#epo = 91
#for i in range(epo):
#    point1 = np.array(A1.points)
#    point2 = np.array(A2.points)
#    p1,p2,indice = pointMatching(point1,point2)
#    print("["+str(i)+"/"+str(epo)+"]loss:" + str(icploss(p1,p2)))
#    Trans = calTransformation(p1,p2)                   #p2p
#    #p2norm_use = p2norm[indice]                         #p2pl
#    #Trans = calTransformation_p2pl(p1, p2,p2norm_use)   #p2pl
#    A1.transform(Trans)
#    if (i in showlist):
#        # o3d.draw_geometries([pcd1+pcd2],window_name="new",width=1920,height=1024)

i = 0

def custom_draw_geometry_with_view_tracking(meshes):
    def track_view(vis):
        global i, A1, A2

        if i % 1 == 0:
            point1 = np.array(A1.points)
            point2 = np.array(A2.points)
            p1, p2, indice = pointMatching(point1,point2)
            #print("["+str(i)+"]loss:" + str(icploss(p1,p2)))
            #Trans = cal_transformation(p1, p2)                   #p2p
            p2norm_use = p2norm[indice]                         #p2pl
            Trans = cal_transformation_p2pl(p1, p2, p2norm_use)   #p2pl
            A1.transform(Trans)
            vis.update_geometry()

        i += 1

    o3d.draw_geometries_with_animation_callback(meshes, track_view)

custom_draw_geometry_with_view_tracking([A1, A2])
