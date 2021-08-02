# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 05:41:33 2021

@author: Ahmed Emad Eldin
"""

import cv2
import numpy as np


def roi(image):
    height , width = image.shape
    triangle = np.array([
                       [(100, height), (475, 325), (width, height)]
                       ])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask,triangle,255)
    mask = cv2.bitwise_and(image,mask)
    return mask

def grey(image):
    #img = np.asarray(image,dtype=np.float32)
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def canny(grayImage):
    edges = cv2.Canny(grayImage,50,150)
    return edges

def gaussianBlur(image):
    return cv2.GaussianBlur(image,(5,5),0)

def display_lines(image,lines):
    #lines_image = np.zeros_like(image)
    
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            
            cv2.line(image,(x1,y1),(x2,y2),(255,0,0),10)
    return image

def make_points(average,image):
    slope,y_int = average
    y1 = image.shape[0]
    
    y2 = int(y1 * (3/5))
    
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    
    return np.array([x1,y1,x2,y2])


def average (lines,image):
    right = []
    left = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        #slope = (y2-y1) / (x2-x1)
        slope = parameters[0]
        y_int = parameters[1]
        #print(parameters)
        #print(slope)
        if slope < 0:
            left.append((slope,y_int))
        else:
            right.append((slope,y_int))
    
    right_avg = np.average(right , axis = 0)
    left_avg = np.average(left,axis = 0)
    
    left_line = make_points(left_avg,image)
    
    right_line = make_points(right_avg,image)
    return np.array([left_line , right_line])
    


if __name__ == '__main__':
    cap = cv2.VideoCapture("./video/test2_v2_Trim.mp4")

        
    while True:
        ret , frame = cap.read()
        
        if not ret:
            print("cant receice from camera ")
            break
        
        gray_image = grey(frame)
        blurred_image = gaussianBlur(gray_image)
        edges = canny(blurred_image)
        isolated = roi(edges)
        
        lines = cv2.HoughLinesP(isolated,2,np.pi/180,100,None,40,5)
        average_lines = average(lines,gray_image)
        lanes = display_lines(frame, average_lines)
        cv2.imshow("lanes",frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        
 
        