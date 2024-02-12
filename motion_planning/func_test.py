#!/usr/bin/env python3

import rclpy # Python Client Library for ROS 2
from motion_planning.find_contour import *
from motion_planning.poly_decomp import *
import numpy as np
def convex_decomp():
    polygon1 = [[0, 0], [5, 0], [2.5, 2.5], [5, 5], [0, 5]]
    polygon = np.array([[0, 0], [5, 0], [5, 5], [2.5, 2.5], [0, 5]])
    polygonMakeCCW(polygon)
    print(polygon)
    # print(polygonQuickDecomp(polygon1))

    return
