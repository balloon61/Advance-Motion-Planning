#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

def convex_polygon_sort(A):
    """If A is a convex polygon, given by a list of points in CCW order
    but not necessarily sorted by increasing angle from the x axis,
    this will return a sorted version for use in convex_minkowski_sum.
    """
    i = np.argmin([v[1] for v in A])
    return A[i:] + A[:i]

def convex_minkowski_sum(A,B):
    """A and B are convex polygons, given by lists of points sorted
    by their angle about the minimum.  The output is a convex polygon
    in the same format.
    """
    aprogress = 0
    bprogress = 0
    res = []
    d = np.array([0,-1])
    res.append(A[aprogress] + B[bprogress])
    while aprogress < len(A) or bprogress < len(B):
        aindex = aprogress%len(A)
        bindex = bprogress%len(B)
        anext = (aprogress + 1)%len(A)
        bnext = (bprogress + 1)%len(B)
        #find minimum angle
        if np.dot(d, A[anext] + B[bindex] - res[-1]) > np.dot(d, A[aindex] + B[bnext] - res[-1]):
            res.append(A[anext] + B[bindex])
            d = res[-1] - res[-2]
            aprogress += 1
        else:
            res.append(A[aindex] + B[bnext])
            d = res[-1] - res[-2]
            bprogress += 1
    return res


def test():
    #triangle
    A = [np.array([-1,0]),np.array([1,0]),np.array([0,1.5])]
    #rectangle
    B = [np.array([3,2]),np.array([6,2]),np.array([6,4]),np.array([3,4])]
    #minkowski sum
    C = convex_minkowski_sum(convex_polygon_sort([-v for v in A]),convex_polygon_sort(B))

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.axis('equal')
    ax1.scatter([0],[0],color='k',label='Reference point')
    ax1.plot(np.array(A+[A[0]])[:,0],np.array(A+[A[0]])[:,1],label='A')
    ax1.plot(np.array(B+[B[0]])[:,0],np.array(B+[B[0]])[:,1],label='B')
    ax1.plot(np.array(C+[C[0]])[:,0],np.array(C+[C[0]])[:,1],label='A - B')
    ax1.legend()

    ax2.set_xlim(1,7)
    ax2.set_ylim(0,6)
    ax2.set_zlim(0,math.pi*2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel(r'$\theta$')
    ax2.view_init(elev=20, azim=45)
    Nslices = 60
    for i in range(Nslices):
        u = float(i)/Nslices
        theta = u*math.pi*2
        R = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        RnA = [-np.dot(R,v) for v in A]
        C = convex_minkowski_sum(convex_polygon_sort(RnA),B)
        #output C at height theta
        tri = PolyCollection([np.array(C)])
        tri.set_color(colors.rgb2hex([0.5+0.5*u,0.5+0.5*u,0.5+0.5*u]))
        tri.set_edgecolor('k')
        ax2.add_collection3d(tri,zs=theta)
        
    plt.show()