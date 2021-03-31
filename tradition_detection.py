# -*- coding: utf-8 -*-

import  cv2
import numpy as np

def is_inside(o,i):
#o和i分别是两个矩形
    ox,oy,ow,oh = o
    ix,iy,iw,ih = i
    return ox > ix and oy > iy and ox+ow < ix+iw and oy+oh < iy+ih
    
def draw_person(img,person):

    x,y,w,h = person
    # cv2.rectangle绘制矩形
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    

def detect_test():

    img = cv2.imread('./vehicle_pictures/car1.jpg')
    rows,cols = img.shape[:2]
    sacle = 1.0
    img = cv2.resize(img,dsize=(int(cols*sacle),int(rows*sacle))) 
    #创建HOG描述符对象
    #计算一个检测窗口特征向量维度：(64/8 - 1)*(128/8 - 1)*4*9 = 3780
    hog = cv2.HOGDescriptor()  
    #hist = hog.compute(img[0:128,0:64])   计算一个检测窗口的维度
    #print(hist.shape)
    detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
    print('detector',type(detector),detector.shape)    
    hog.setSVMDetector(detector)

    #多尺度检测，found是一个数组，每一个元素都是对应一个矩形，即检测到的目标框
    found,w = hog.detectMultiScale(img)
    print('found',type(found),found.shape)
    
    #过滤一些矩形，如果矩形o在矩形i中，则过滤掉o
    found_filtered = []
    for ri,r in enumerate(found):
        for qi,q in enumerate(found):
            #r在q内？
            if ri != qi and is_inside(r,q):
                break
        else:
            found_filtered.append(r)
            
    for person in found_filtered:
        draw_person(img,person)
        
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    detect_test()