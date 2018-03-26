# imports
from helper_functions import *
# ---------------------------------

C_DEBUG = False


def camera_calibration(images,chess_corners=(4,4)):
    # do the calibration, return matrices etc

    shape = images[0].shape;

    for im in images:
        image_points = []
        object_points = []

        nx=chess_corners[0]
        ny=chess_corners[1]

        # fill object points
        objp = np.zeros((nx*ny,3),np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)


        ret, corners = cv2.findChessboardCorners(gray(im),chess_corners,None)
        if ret == True:
            image_points.append(corners)
            object_points.append(objp)

        if C_DEBUG:
            print('--' + str(ret))
            cv2.drawChessboardCorners(im,chess_corners,corners,ret)
            show_image(im)

    ret, mtx, distortion, rot_vect, trans_vect = cv2.calibrateCamera(object_points,image_points,(shape[0],shape[1]),None,None)

    if C_DEBUG:
        print('------- camera_calibration --------')
        print((ret, mtx, distortion, rot_vect, trans_vect))

    return (ret, mtx, distortion, rot_vect, trans_vect)

def undistort_img(img,mtx,dist):
    return cv2.undistort(img,mtx,dist,None,mtx)



def image_processing(img,Tmtrx):
    #process image and return an image for further processing

    #perspective
    pers = perspective(img,Tmtrx,(640,1000))


    img_gray = cv2.cvtColor(pers,cv2.COLOR_RGB2GRAY)

    img_hsv = cv2.cvtColor(pers, cv2.COLOR_RGB2HSV)

    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)
    img_sobel_x = np.absolute(img_sobel_x)
    img_sobel_x = np.uint8(255 * img_sobel_x / np.max(img_sobel_x))

    bin_hsv = binFilter(img_hsv,[(0,255),(100,255),(50,255)])





    if C_DEBUG:
        print('debug')
        show_image(pers)
        show_image(img_gray)
        show_image(img_hsv)
        show_image(img_sobel_x)
        show_image(bin_hsv)


    return img


def extract_lane_single(img):


    return [left,right]

def test():
    #img = load_image('test_images/straight_lines1.jpg')
    #show_image(img)
    imgList=load_images('camera_cal/')
    show_image(imgList[3])



def main():
    # main, do something
    global C_DEBUG # may be changed here


    # calibrate camera
    cal_chess_corners = (9,6)
    cal_images = load_images('camera_cal/')
    t1, mtx, distortion, t2, t3 = camera_calibration(cal_images,cal_chess_corners)

    C_DEBUG = False
    if C_DEBUG:
        for im in cal_images:
            iu = undistort_img(im,mtx,distortion)
            show_image(iu)



    # examples
    ##C_DEBUG = True
    images = load_images('test_images/')
    if C_DEBUG:
        for im in images:
            iu = undistort_img(im,mtx,distortion)
            show_image(iu)

    src  = np.float32([(220,720), (570,470), (720,470), (1110,720)])
    dest = np.float32([(220,2000), (220,0), (1110,0), (1110,2000)])
    dest *= 0.5
    Tmtrx=getTransformMatrix(src,dest)


    C_DEBUG = True
    for im in images:
        image_processing(undistort_img(im,mtx,distortion),Tmtrx)



    #load video / sample images





    #process images with calibration

    #detect lines


    return 0


if __name__ == '__main__':
    main()