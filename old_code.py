##############################################################################################
# this is just the code from the first line finding project
# copied from the 'notebook' in case i might want to reuse something of it
##############################################################################################


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return [line_img, lines]


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

import os
os.listdir("test_images/")

import os
import sys


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

def order_lines(lines, x_middle, bExcludeWrongSlope=False, maxSlope=2):
    lines_left = []
    lines_right = []
    for line in lines:
        bIsLeft = True
        for x1, y1, x2, y2 in line:
            if (x1 + x2) / 2 > x_middle:  # is the average of the lines x-components in the right half?
                bIsLeft = False
        if bExcludeWrongSlope:
            dy = y2 - y1
            dx = x2 - x1
            if dy == 0:
                continue
            m = (dx) / (dy)
            if bIsLeft:
                if m > 0:
                    continue
            else:
                if m < 0:
                    continue
            if abs(m) > maxSlope:
                continue

        if bIsLeft:
            lines_left.append(line)
        else:
            lines_right.append(line)

    return [lines_left, lines_right]


def average_lineparams(lines):
    # get average x=my+b parameters, weighted by square of the length  # note: switched to y beeing run variable

    if len(lines) < 1:
        return [0, 0]

    m = 0
    b = 0
    w = 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            dy = y2 - y1
            dx = x2 - x1
            if dy == 0:
                continue
            m_tmp = (dx) / (dy)
            b_tmp = x1 - m_tmp * y1
            # print(b_tmp)
            # w_tmp = math.sqrt(math.pow(dx, 2) + math.pow(dy,2))
            w_tmp = math.pow(dx, 2) + math.pow(dy, 2)
            # w_tmp = 1
            m = m + (m_tmp * w_tmp)
            b = b + (b_tmp * w_tmp)
            w = w + w_tmp

    m = m / w
    b = b / w
    return [m, b]


def write_image(image, name):
    outPath = 'pipeline_steps_images/'
    if len(image.shape) == 3:
        cv2.imwrite(outPath + name + '.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite(outPath + name + '.png', image)


def process(image, num=1):
    dx = image.shape[1]
    dy = image.shape[0]

    # params

    # debug
    bDebug = False
    bWriteImages = False
    # bluring
    bluring_kernel = 5
    # canny
    canny_low = 50
    canny_high = 150
    # margins
    mx_bottom = math.floor(0 * dx)
    mx_top = math.floor(0.45 * dx)
    my_top = math.floor(0.6 * dy)
    if bDebug:
        print(mx_bottom, mx_top, my_top)

    # hough
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 11  # # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # #minimum number of pixels making up a line
    max_line_gap = 3  # maximum gap in pixels between connectable line segments

    # printing out some stats and plotting
    if bDebug:
        print('This image is:', type(image), 'with dimensions:', image.shape)

    dx_half = math.floor(dx / 2)

    imWork = image

    if bWriteImages:
        write_image(imWork, '0_input')

    # grayscale
    imWork = grayscale(image)

    if bWriteImages:
        write_image(imWork, '1_gray')

    # blur
    imWork = gaussian_blur(imWork, bluring_kernel)

    if bWriteImages:
        write_image(imWork, '2_blur')

        # canny
    imWork = canny(imWork, canny_low, canny_high)

    if bWriteImages:
        write_image(imWork, '3_cany')

    # selection mask
    mask = np.zeros_like(imWork)
    ignore_mask_color = 255

    # select region
    vertices = np.array(
        [[(0 + mx_bottom, dy), (0 + mx_top, 0 + my_top), (dx - mx_top, 0 + my_top), (dx - mx_bottom, dy)]],
        dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # ignore rest
    imWork = cv2.bitwise_and(imWork, mask)

    if bWriteImages:
        write_image(imWork, '4_masked')

    # hough
    [imWork, lines] = hough_lines(imWork, rho, theta, threshold, min_line_length, max_line_gap)
    # [imDummy, lines] = hough_lines(imWork, rho, theta, threshold, min_line_length, max_line_gap)
    imageOut = imWork

    if bWriteImages:
        write_image(imWork, '5_hough')

    [ll, lr] = order_lines(lines, dx_half, True)
    if bDebug:
        print(len(ll))
        print(len(lr))
        print(len(lines))

        draw_lines(imageOut, ll, [0, 255, 0], 2)
        draw_lines(imageOut, lr, [255, 0, 0], 2)

    if bWriteImages:
        tmp = imWork
        draw_lines(tmp, ll, [0, 255, 0], 2)
        draw_lines(tmp, lr, [255, 0, 0], 2)
        write_image(tmp, '6_lines_left_right')

    bDrawLeft = len(ll) > 0
    bDrawRight = len(lr) > 0

    [m_l, b_l] = average_lineparams(ll)
    [m_r, b_r] = average_lineparams(lr)

    #   try:
    avg_left_line = [math.floor(m_l * my_top + b_l), my_top, math.floor(m_l * dy + b_l), dy]
    #  except:
    #      print('Exception in left lane:')#+sys.exc_info()[0])
    #      bDrawLeft=False


    #  try:
    avg_right_line = [math.floor(m_r * my_top + b_r), my_top, math.floor(m_r * dy + b_r), dy]
    #  except:
    #      print('Exception in right lane:')#+sys.exc_info()[0])
    #      bDrawRight=False


    if bDebug:
        draw_lines(imageOut, [[avg_left_line]], [0, 255, 128], 2)
        draw_lines(imageOut, [[avg_right_line]], [255, 0, 128], 2)

    # print(lines)

    if bDebug:
        plt.figure(num + 1)
        plt.imshow(image)
        plt.figure(num + 2)

        # middle line
        cv2.line(imageOut, (dx_half, 0), (dx_half, dy), [0, 0, 255], 3)

        plt.imshow(imageOut,
                   cmap='gray')  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

    black = np.zeros_like(imWork)

    if bDebug:
        black = imageOut

    if bDrawLeft:
        draw_lines(black, [[avg_left_line]], [0, 255, 0], 10)
    if bDrawRight:
        draw_lines(black, [[avg_right_line]], [255, 0, 0], 10)

    result = weighted_img(black, image)

    if bWriteImages:
        write_image(black, '7_single_lines')
        write_image(result, '8_result')

    return result


imageList = os.listdir("test_images/")
# print(imageList)
figNum = 1
# for filename in imageList:
filename = imageList[2]
image = mpimg.imread("test_images/" + filename)
res = process(image, figNum)
plt.figure(figNum)
plt.imshow(res)
figNum = figNum + 3

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = process(image)

    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)



challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
%time challenge_clip.write_videofile(challenge_output, audio=False)



