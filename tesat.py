#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/6/27 9:19
#@Author: csc
#@File  : tesat.py

from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array([[1,2,3],[2,3,4],[5,6,7]])
Y = np.array([[2.1,3.1,4.1],[5.1,6.1,7.1],[1.1,2.1,3.1]])
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(Y)
print("index")
print(indices)
print("dist")
print(distances)