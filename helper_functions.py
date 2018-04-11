# helper functions for lane finding.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from moviepy.editor import VideoFileClip
from functools import partial

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
    fullValue = 255
    try:
        binary = np.ones_like(img[:,:,1]) # shape!!
    except:
        binary = np.ones_like(img)  # shape!!

    binary *= fullValue
    #binary[0,0]=0 # one black pixel for visualisation

    #show_image(binary,True)

    for i in range(len(thres)):
        if len(thres) > 1:
            tmp = img[:,:,i]
        else:
            tmp = img
        #show_image(tmp, True)
        thrs = thres[i]
        binary[(tmp < thrs[0]) | (tmp > thrs[1]) | (binary < 1)] = 0
        #show_image(binary, True)

    return binary

def comBinary(imges,bAnd=True):
    if bAnd:
        ret = np.ones_like(imges[0])
        for i in range(len(imges)):
            ret[(ret < 1) | (imges[i] < 1)] = 0
        return ret
    else: #OR
        ret = np.zeros_like(imges[0])
        for i in range(len(imges)):
            ret[(ret > 0) | (imges[i] > 0)] = 1
        return ret

def show_image(image,bBW=False):
    if bBW:
        plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image)
    plt.show(block=True)

class Window:
    def __init__(self,tl,br,np=0,bValid=False):
        self.tl = tl
        self.br = br
        self.num_pixels = np
        self.valid = bValid
        self.yMean = (br[0]-tl[0])/2
        self.xMean = (br[1]-tl[1])/2
        self.weight = 0

    def set_x_center(self,val):
        dx = int(val) - self.get_x_center()
        self.tl = (self.tl[0],int(self.tl[1]+dx))
        self.br = (self.br[0], int(self.br[1] + dx))
        self.xMean = (self.br[1]-self.tl[1])/2;
        return dx
        #print("Set: "+str(val)+"  "+str((self.br[1] + self.tl[1]) / 2))

    def get_x_center(self):
        cx= int((self.br[1] + self.tl[1]) / 2)
        #print("Get: " + str(cx))
        return cx

    def get_y_center(self):
        return int((self.br[0]+self.tl[0])/2)

    def get_y_mean(self):
        return int(self.yMean)

    def get_y_mean_total(self):
        return self.tl[0]+self.get_y_mean()

    def get_x_mean(self):
        return int(self.xMean)

    def get_x_mean_total(self):
        return self.tl[1]+self.get_x_mean()

    def get_weight(self):
        return int(self.weight)


    def cropImg(self,img):
        yMin=max(self.tl[0],0)
        yMax=min(self.br[0],img.shape[0])
        xMin=max(self.tl[1],0)
        xMax=min(self.br[1],img.shape[1])
        return img[yMin:yMax, xMin:xMax]


    def centerAroundPixels(self,img,min_px,lastweight,lastx,lasty):
        w_img = self.cropImg(img)
        #show_image(w_img)
        pixels = w_img.nonzero()
        total_pixels = w_img.shape[0]*w_img.shape[1]

        pxcount = len(pixels[1])
        alpha_last = 0
        if (lastweight+pxcount) > 0:
            alpha_last = lastweight/(lastweight+pxcount)

        # enough pixels?

        #print(pxcount)
        if pxcount > min_px and pxcount < (0.2 * total_pixels):
            mean_x = int(np.mean(pixels[1])*(1-alpha_last)+lastx*(alpha_last))
            mean_y = int(np.mean(pixels[0])*(1-alpha_last)+lasty*(alpha_last))

           # print(mean)
            dx=self.set_x_center(self.tl[1] + mean_x)
            self.yMean = mean_y
            ##self.xMean = mean_x-dx
            self.valid = True
            self.weight=int(pxcount*(1-alpha_last) + lastweight * alpha_last)
            #print("Valid")
        else:
            self.valid = False


            #print("-- Nope --")
        #print (self.tl)


def createWindows(in_width,in_height,count_y,width,bRight=False):
    # creates windows for left and right starting at the botoom
    list = []
    im_width=in_width
    w_height = in_height/count_y
    offset_left = 0
    if bRight:
        offset_left = im_width-width
    for i in reversed(range(count_y)):
        list.append(Window((i*w_height,offset_left),((i+1)*w_height,offset_left+width),0,False))
    return list



def find_windows(img,WindowsLeft,WindowsRight,minPixels):
    bottom_part_of_image = 0.5
    size_x = img.shape[1]
    size_y = img.shape[0]

    # get histogram for start of lanes (bottom window) (sum over y at each x)
    hist = np.sum(img[int((1-bottom_part_of_image)*size_y):,:], axis = 0)

    #plt.plot(hist)
    #plt.show()

    LR_split = (size_x//2)
    xLeft = np.argmax(hist[:LR_split])
    xRight  = np.argmax(hist[LR_split:]) + LR_split

    for i in range(len(WindowsLeft)):

        # use last frames values if it was valid available
        # otherwise the window below

        #left
        if (not WindowsLeft[i].valid) and (WindowsLeft[i].get_weight() < 100 ): # last frame valid?
            if i==0:
                WindowsLeft[i].set_x_center(xLeft) # to histogram peak
            else:
                cx = WindowsLeft[i-1].get_x_mean_total() # to window below
                WindowsLeft[i].set_x_center(cx)

        #right
        if (not WindowsRight[i].valid) and (WindowsRight[i].get_weight() < 100 ):
            if i == 0:
                WindowsRight[i].set_x_center(xRight)
            else:
                cx = WindowsRight[i - 1].get_x_mean_total()
                WindowsRight[i].set_x_center(cx)

        WindowsLeft[i].centerAroundPixels(img, minPixels,WindowsLeft[i].get_weight(),WindowsLeft[i].get_x_mean(),WindowsLeft[i].get_y_mean())
        WindowsRight[i].centerAroundPixels(img, minPixels,WindowsRight[i].get_weight(),WindowsRight[i].get_x_mean(),WindowsRight[i].get_y_mean())

    return hist

def fit_windows(windows,xRatio,yRatio):
    pointsX = []
    pointsY = []
    for w in windows:
        if w.valid:
            for px in range(w.get_weight()):
                pointsX.append(w.get_x_mean_total()*xRatio)
                pointsY.append(w.get_y_mean_total()*yRatio)

    func_in_y = np.polyfit(pointsY,pointsX,2)
    return func_in_y  #windows[0].get_x_center() # polynom and start 'x'

def plot_poly(poly, out_size):
    y = np.linspace(0, out_size[1] - 1, out_size[1])
    x = poly[0] * y ** 2 + poly[1] * y + poly[2]

    line = np.array([np.transpose(np.vstack([x, y]))])

    return line #, y, x



