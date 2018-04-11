# imports
from helper_functions import *
# ---------------------------------

C_DEBUG = False
VID_DEBUG = True


def camera_calibration(images,chess_corners=(4,4)):
    # do the calibration, return matrices etc

    shape = images[0].shape

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


def image_processing(img,Tmtrx,out_size):
    #process image and return an image for further processing

    #perspective
    pers = perspective(img,Tmtrx,out_size) # perspective warp first

    # prepare color spaces
    img_gray = cv2.cvtColor(pers,cv2.COLOR_RGB2GRAY)
    img_hls = cv2.cvtColor(pers, cv2.COLOR_RGB2HLS)

    # edge detection
    img_sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0) # only in X (perspective already 'corrected')
    img_sobel = np.absolute(img_sobel)
    img_sobel = np.uint8(255 * img_sobel / np.max(img_sobel)) # 0..255

    # binary images
    # lanes by color
    bin_lanes = binFilter(img_hls, [(0, 60), (80, 255), (110, 255)])
    # lanes by edge detection
    bin_sobel = binFilter(img_sobel,[(50,255)])

    # OR-combine
    bin_img = comBinary([bin_lanes, bin_sobel],False)

    if C_DEBUG:
        print('debug')
        show_image(pers)
        show_image(bin_lanes)
        show_image(bin_sobel)
        show_image(bin_img)

    return bin_img, pers


def test():
    #img = load_image('test_images/straight_lines1.jpg')
    #show_image(img)
    imgList=load_images('camera_cal/')
    show_image(imgList[3])

def process_frame(mtx,distortion,Tmtrx,Tmtrx_Inv,out_size,windowsLeft,windowsRight,im):
    # process images to binary lines
    im_undist = undistort_img(im, mtx, distortion)

    im_bin, pers= image_processing(im_undist, Tmtrx, out_size)

    # find windows around lines
    hist = find_windows(im_bin,windowsLeft,windowsRight,100)

    xRatio = 1
    yRatio = 1

    poly_left = fit_windows(windowsLeft,xRatio,yRatio)
    line_l = plot_poly(poly_left,out_size)

    poly_right = fit_windows(windowsRight, xRatio, yRatio)
    line_r = plot_poly(poly_right, out_size)

    area = np.hstack((line_l,np.fliplr(line_r)))

    lane_img = np.zeros_like(pers)
    cv2.fillPoly(lane_img, np.int32(area), color=(0, 255, 255))
    cv2.polylines(lane_img, np.int32(line_l), isClosed=False, color=(0, 0, 255), thickness=10)
    cv2.polylines(lane_img, np.int32(line_r), isClosed=False, color=(0, 0, 255), thickness=10)


    if C_DEBUG or VID_DEBUG:
#    if 1:
        # draw lines
        im_out = np.dstack((im_bin, im_bin, im_bin))*255
        all_windows = []
        all_windows.extend(windowsLeft)
        all_windows.extend(windowsRight)
        for w in all_windows:
            if w.valid:
                cv2.rectangle(im_out, (w.tl[1],w.tl[0]), (w.br[1],w.br[0]), (0, 255, 0), 5)
            else:
                cv2.rectangle(im_out, (w.tl[1], w.tl[0]), (w.br[1], w.br[0]), (255,0, 0), 5)



        im_out = weighted_img(pers,im_out,0.9)
        im_out = weighted_img(im_out,lane_img, 0.8)

        lane_img = im_out
        ##show_image(im_out)

    unwarp = perspective(lane_img,Tmtrx_Inv,(im_undist.shape[1],im_undist.shape[0]))

    processed_img = weighted_img(unwarp,im_undist,1,0.5,0)


    #get curvature
    world_yRatio = 30.0/(out_size[1]) #m/px
    world_xRatio = 3.7/(out_size[0]*(455.0/600.0)) #m/px

    poly_left = fit_windows(windowsLeft, world_xRatio, world_yRatio)
    poly_right = fit_windows(windowsRight, world_xRatio, world_yRatio)

    left_rad = ((1 + (2*poly_left[0]*out_size[1]*world_yRatio + poly_left[1])**2)**1.5) / np.absolute(2*poly_left[0])
    right_rad = ((1 + (2*poly_right[0]*out_size[1]*world_yRatio + poly_right[1])**2)**1.5) / np.absolute(2*poly_right[0])

    x_left_world = poly_left[0] * out_size[1]*world_yRatio** 2 + poly_left[1] * out_size[1]*world_yRatio + poly_left[2]
    x_right_world = poly_right[0] * out_size[1]*world_yRatio** 2 + poly_right[1] * out_size[1]*world_yRatio + poly_right[2]

    car_x = out_size[0]*world_xRatio / 2
    lane_center = (x_right_world-x_left_world)/2

    offset = lane_center-car_x

    cv2.putText(processed_img, 'Curve Radius: ' + str((left_rad + right_rad) / 2)[:7] + ' m', (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(processed_img, 'Offset: ' + str((offset))[:7] + ' m', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
    #print(left_rad, right_rad, offset)

    ##show_image(processed_img)
    return processed_img


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
    #C_DEBUG = True
    images = load_images('test_images/')


    if C_DEBUG:
        for im in images:
            iu = undistort_img(im,mtx,distortion)
            show_image(iu)

    src  = np.float32([(220,720), (570,470), (720,470), (1110,720)])
    dest = np.float32([(220,2000), (220,0), (1110,0), (1110,2000)])
    dest *= 0.5 # make it a bit smaller
    Tmtrx=getTransformMatrix(src,dest)
    Tmtrx_Inv = getTransformMatrix(dest,src)

    out_size = (640, 1000) # x,y
    window_y_count = 10
    window_width = 150

    windowsLeft = createWindows(out_size[0],out_size[1],window_y_count,window_width,False)
    windowsRight= createWindows(out_size[0],out_size[1],window_y_count,window_width,True)

    ## Preparation Done ---- Looping images / video from here on
    Video=True
    #C_DEBUG=True

    if Video:
        project_video_output = "project_video_output_2.mp4"
        project_video_input = VideoFileClip("project_video.mp4")

        #define partial function for arguments

        video_frame = partial(process_frame,mtx, distortion, Tmtrx, Tmtrx_Inv, out_size, windowsLeft, windowsRight)

        processed_project_video = project_video_input.fl_image(video_frame)
        processed_project_video.write_videofile(project_video_output, audio=False)

    else:
        for im in images:
            res = process_frame(mtx,distortion,Tmtrx,Tmtrx_Inv,out_size, windowsLeft,windowsRight,im)
            show_image(res)

    return 0


if __name__ == '__main__':
    main()