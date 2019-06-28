#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/26 6:00
#@Author: csc
#@File  : icp.py

from icp_support import *

pcd1 = o3d.io.read_point_cloud("./OralScans/0.ply")
pcd2 = o3d.io.read_point_cloud("./OralScans/1.ply")
o3d.visualization.draw_geometries([pcd1+pcd2],window_name="old",width=300,height=300)
showlist = [1,20, 50, 90]
epo = 91
for i in range(epo):
    point1 = np.array(pcd1.points)
    point2 = np.array(pcd2.points)
    p1,p2 = pointMatching(point1,point2)
    print("["+str(i)+"/"+str(epo)+"]loss:" + str(icploss(p1,p2)))
    Trans = calTransformation(p1,p2)
    pcd1.transform(Trans)
    if(i in showlist):
        o3d.visualization.draw_geometries([pcd1+pcd2],window_name="new",width=300,height=300)


