# helper functions for lane finding.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def draw_lines(img, lines, color=[255, 0, 0], thickness=2): # as from the first project
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):   # as from the first project
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def gray(img_in,conversion=cv2.COLOR_RGB2GRAY): # use COLOR_BGR2GRAY if loaded by cv2
    return cv2.cvtColor(img_in, conversion)


def write_image(image, path):
    if len(image.shape) == 3:
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite(path, image)


def load_images(folder):
    fileList = os.listdir(folder)
    imgList = []
    for f in fileList:
        imgList.append(load_image(folder+f))
    return imgList

def getTransformMatrix(sources, targets):
    return cv2.getPerspectiveTransform(sources,targets)

def perspective(im,matrix,dest_size):
    return cv2.warpPerspective(im,matrix,dest_size,flags=cv2.INTER_LINEAR)

def load_image(path):
    return mpimg.imread(path)

def binFilter(img,thres): # multi channel binary-AND-Filter
    binary = np.ones_like(img) # shape!!

    for i in range(len(thres)):
        tmp = img[:,:,i]
        thrs = thres[i]
        binary[(tmp >= thrs[0]) & (tmp <= thrs[1]) & (binary > 0)] = 1

def comBinary(imges):
    ret = np.ones_like(imges[0])
    for i in range(len(imges)):
        ret[(ret > 0) & (imges[i] > 0)] = 1
    return ret

def show_image(image):
    plt.imshow(image)
    plt.show(block=True)



