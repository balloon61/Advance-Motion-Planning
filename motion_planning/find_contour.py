#!/usr/bin/env python3

import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

import numpy as np



def find_contour():

    # read the image
    image = cv2.imread('map.png')
    
    # apply binary thresholding
    ret, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    # visualize the binary image
    cv2.imshow('Binary image', thresh)
    cv2.waitKey(0)
    cv2.imwrite('image_thres1.jpg', thresh)
    cv2.destroyAllWindows()

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                        
    # draw contours on the original image
    image_copy = image.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    
    # see the results
    cv2.imshow('None approximation', image_copy)
    cv2.waitKey(0)
    cv2.imwrite('contours_none_image1.jpg', image_copy)
    cv2.destroyAllWindows()


# def get_kernel(size: np.ndarray, ):



def Minkowski_sum():
    image = cv2.imread('map.png')
