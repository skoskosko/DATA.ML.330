import cv2
import numpy as np
import random
import os
import glob

def apply_brightness_contrast(input_img, brightness = 110, contrast = 200):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def get_spots(filename):
    response = []
    image = cv2.imread(filename, 0) 
    imageColor = cv2.imread(filename)
    s1= 100 # min countour size
    s2 = 10000 # max countour size
    index = 0 # index of saved image


    th, threshed = cv2.threshold(image, 100, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    xcnts = []
    for cnt in cnts:
        if s1<cv2.contourArea(cnt) <s2:
            xcnts.append(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            response.append({"xmin" : x , "ymin" : y, "xmax" : x+w, "ymax": y+h})
            
    largest_contour = max(cnts, key = cv2.contourArea)
    contr_image = apply_brightness_contrast(image)

    th, threshed = cv2.threshold(contr_image, 100, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    count_before_contrast=len(xcnts)
    for cnt in cnts:
        if s1<cv2.contourArea(cnt) <s2:
            xcnts.append(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            response.append({"xmin" : x , "ymin" : y, "xmax" : x+w, "ymax": y+h})
   
    height, width = image.shape
    

    return response, height, width


textfile = "filename,width,height,class,xmin,ymin,xmax,ymax\n"

for index in range(2578):
    filename = "./Dry/images/dry_%s.png" % str(index)
    items, height, width = get_spots(filename)

    for item in items:
        textfile += "dry_{id}.png,{width},{height},branch,{xmin},{ymin},{xmax},{ymax}\n".format(id=index, width=str(width), height=str(height), xmin=item["xmin"], ymin=item["ymin"], xmax=item["xmax"], ymax=item["ymax"])
    

with open("./Dry/labels.csv", "w") as file:
    file.write(textfile)
