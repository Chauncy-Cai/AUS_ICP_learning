#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/27 3:26
#@Author: csc
#@File  : icp_test.py

from icp_support import *
from TransformGeneration import *
import numpy as np
import open3d as o3d

A1 = o3d.geometry.TriangleMesh.creat_box(1,2,3)
#似乎版本不对，不存在create box函数？
Tran = transform(0.2, 0.3, 0.4, 1, 2, 3)
A2 = A1.tranform(Tran)
o3d.visualization.draw_geometries([A1],window_name="old")
o3d.visualization.draw_geometries([A2],window_name="new")