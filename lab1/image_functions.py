import os
import numpy as np
from skimage import io, morphology
import cv2

def image_processing(image_path):

    train_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    norm_img= cv2.normalize(train_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    canny = cv2.Canny(norm_img,80,135)
    
    kernel = np.ones((3,3))
    dilated = cv2.dilate(canny, kernel) 
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    arr = closed > 0
    cleaned = morphology.remove_small_objects(arr, min_size=100)
    cleaned = morphology.remove_small_holes(cleaned, 100)
    
    return cleaned * 255

def find_figures_contours(image):

    contours, hierarchy = cv2.findContours(image.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    l = []
    for c in contours:
        if cv2.contourArea(c) > 1000 and cv2.contourArea(c) < 5000:
            l.append(c)
    
    a = np.zeros(image.shape)
    for i in l:
        for j in i:
            for z in j:
                a[z[1], z[0]] = 255
    a = a.astype(np.uint8)
    
    contours, hierarchy = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def find_card_contours(image):
  
    contours, hierarchy = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def smooth(contours):
    answer_list = []
    for cnt in contours:
        max_angle = 3.14
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        app = approx.reshape((1, -1, 2))[0]
        for a in app:
            for b in app:
                if a[0]!=b[0] and a[1]!=b[1]:
                    m1, m2 = np.sqrt(a[0]**2 + a[1]**2), np.sqrt(b[0]**2 + b[1]**2)
                    angl = np.arccos((a[0]*b[0] + a[1]*b[1]) / (m1 * m2))
                    if angl > max_angle:
                        max_angle = angl
        answer_list.append(len(approx) > 9 and max_angle > 0.1)
    return answer_list

def convex(contours):
    answer1_list, answer_dict = [], {True: 'C', False: ''}
    for cnt in contours:
        epsilon = 0.02*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        answer1_list.append(answer_dict[cv2.isContourConvex(approx)])
    return answer1_list

def numbers(contours):
    answer2_list = []
    for cnt in contours:
        epsilon = 0.02*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        answer2_list.append(str(len(approx)))
    return answer2_list

def expert_image_processing(image_path):
    
    train_image = cv2.imread(image_path)  
    hsv = cv2.cvtColor(train_image, cv2.COLOR_BGR2HSV) 
    max_value = 255
    max_value_H = 130
    low_H = 80
    low_S = 0
    low_V = 0
    high_H = max_value_H
    high_S = max_value
    high_V = max_value
    img_g = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    
    for i in range(3):
        hsv[..., i] *= img_g    
    res = cv2.cvtColor(hsv, cv2.IMREAD_GRAYSCALE)[..., 1]
    for i in range(3, 4):
        res += cv2.cvtColor(hsv, cv2.IMREAD_GRAYSCALE)[..., i]

    norm_img= cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    canny = cv2.Canny(norm_img,80,135)
    
    kernel = np.ones((3,3))
    dilated = cv2.dilate(canny, kernel) 
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    arr = closed > 0
    cleaned = morphology.remove_small_objects(arr, min_size=100)
    cleaned = morphology.remove_small_holes(cleaned, 100)
    
    return closed

def beg_marks(contours, train_image):
    i = 0
    cv2.drawContours(train_image, contours,-1,(0,255,0),3)
    for cnt in contours:
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        org = cX, cY
        color = (0, 255, 255)
        thickness = 2
        i += 1
        train_image = cv2.putText(train_image, '#' + str(i), org, cv2.FONT_HERSHEY_SIMPLEX,
                                     1, color, thickness, cv2.LINE_AA)
    return train_image

def markup(contours, train_image):
    lst1, lst2, lst3 = smooth(contours), convex(contours), numbers(contours)
    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        org = cX, cY
        color = (0, 0, 255)
        thickness = 2

        if lst1[i]:
            train_image = cv2.putText(train_image, 'S', org, cv2.FONT_HERSHEY_SIMPLEX,
                                     1, color, thickness, cv2.LINE_AA)
        else:
            train_image = cv2.putText(train_image, 'P' + lst3[i] + lst2[i], org, cv2.FONT_HERSHEY_SIMPLEX,
                                     1, color, thickness, cv2.LINE_AA)
    return train_image