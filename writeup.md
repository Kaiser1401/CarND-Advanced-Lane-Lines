## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)

[img_cal]: ./output_images/i_40.png "Distorted"
[img_cal2]: ./output_images/i_41.png "Undistorted"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

This.

### Code Files
The code for the project is written in two files:

* `find_lanes.py` Entry main function, camera calibration, undistortion, image/frame processing
* `helper_functions.py`All the functions needed for that

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for `camera_calibration()` is called from `main()` in `find_lanes.py`

Object points are prepared to represent x,y,z coordinates of the corners in the world. The chessboard is assumed to be in a plane with fixed x and y and z=0. The corners in each calibration image get assigned to the same world coordinates.  Thus, `objp` is just a replicated array of coordinates, and `object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image with `cv2.findChessboardCorners()`.  `image_points` will be appended with the x and y pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `object_points` and `image_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result which also shows the detected corners found by `cv2.findChessboardCorners`: 

| Dirstorted        | Undistorted  | 
|:-------------:|:-------------:| 
| ![][img_cal]  | ![][img_cal2] | 

### Processing Pipeline

The pipeline is basically the same for image processing and video processing.

For videoprocessing a partial function thats fills all arguments but the frame has been declared to be called. See `find_lanes.py:~260`

For single picture processing a variable `_RESET_ALLWAYS_` has been introduced and is set to `True` to not use the last output as guidance for the next.

Before calling `process_frame()` the calibration data is generated `find_lanes.py:~217` and passed along with reusable  `windows` objects `find_lanes.py:~245` for left and right lane.

The steps for each picture / frame are as follows (see descriptions below):

[i60]: ./output_images/i_60.png ""
[i61]: ./output_images/i_61.png ""
[i62]: ./output_images/i_62.png ""
[i63]: ./output_images/i_63.png ""
[i64]: ./output_images/i_64.png ""
[i65]: ./output_images/i_65.png ""
[i66]: ./output_images/i_66.png ""
[i67]: ./output_images/i_67.png ""
[i68]: ./output_images/i_68.png ""

[i69]: ./output_images/i_69.png ""
[i70]: ./output_images/i_70.png ""
[i71]: ./output_images/i_71.png ""
[i72]: ./output_images/i_72.png ""
[i73]: ./output_images/i_73.png ""
[i74]: ./output_images/i_74.png ""
[i75]: ./output_images/i_75.png ""
[i76]: ./output_images/i_76.png ""
[i77]: ./output_images/i_77.png ""

[i123]: ./output_images/i_123.png ""
[i124]: ./output_images/i_124.png ""
[i125]: ./output_images/i_125.png ""
[i126]: ./output_images/i_126.png ""
[i127]: ./output_images/i_127.png ""
[i128]: ./output_images/i_128.png ""
[i129]: ./output_images/i_129.png ""
[i130]: ./output_images/i_130.png ""
[i131]: ./output_images/i_131.png ""

| Step | Ex1 | Ex2 | Ex3 |
|:--:|:--:|:--:|:--:|
| 0 Input |![][i60]|![][i69]|![][i123]|
| 1 Undistort|![][i61]|![][i70]|![][i124]|
| 2 Perspective|![][i62]|![][i71]|![][i125]|
| 3 Lanes (Color)|![][i63]|![][i72]|![][i126]|
| 4 Lanes (x-Edges)|![][i64]|![][i73]|![][i127]|
| 5 Combined|![][i65]|![][i74]|![][i128]|
| 6 Histogram|![][i66]|![][i75]|![][i129]|
| 7 Windows & Fitting|![][i67]|![][i76]|![][i130]|
| 8 Result ||![][i68]|![][i77]|![][i131]|



#### 1. Undistort image
With the calibration info the frame is undistorted in `find_lanes.py:~99`

#### 2. Perspective (Birds-Eye)

For perspective transformation `cv2.warpPerspective()` is called through `image_processing()`->`perspective()` in `find_lanes.py:~56` to return an image of size 640 X 1000 pixels.

The transformation matrix is calculated in `find_lanes.py:~235` with the following source and destintion points:

```python
    src  = np.float32([(220,720), (570,470), (720,470), (1110,720)])
    dest = np.float32([(220,2000), (220,0), (1110,0), (1110,2000)])
    dest *= 0.5 # make it a bit smaller
    Tmtrx=getTransformMatrix(src,dest)
    Tmtrx_Inv = getTransformMatrix(dest,src)
```

The example images in the table above show that on straight roads lane markings are parallel to the image borders.

Perspective correction is applied before Lane detection in order to reduce the later edge detection to edges in x-direction only.

#### 3. Lane Detection (Color)

To detect lines by color, the frame is converted to HLS color space in `image_processing()` wich allows filtering by hue (0..60), lightnes (80..255) and saturation (110..255) instead of RGB values to create a binary image of the yellow and white markings.

`find_lanes.py:~60`:
```python
	img_hls = cv2.cvtColor(pers, cv2.COLOR_RGB2HLS)
	# ...
	# lanes by color
	bin_lanes = binFilter(img_hls, [(0, 60), (80, 255), (110, 255)])
```

#### 4. Lane Detection (x-Edges)

Additionally Lanes are detected by edges along the x-axes only, as the image is already in birds view and the interesting edges should all be along this axis. Detection is based on a gray scale image and the Sobel algorithm

`find_lanes.py:~59`:
```python
	img_gray = cv2.cvtColor(pers,cv2.COLOR_RGB2GRAY)
	# ...
	# edge detection
	img_sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0) # only in X (perspective already 'corrected')
	img_sobel = np.absolute(img_sobel)
	img_sobel = np.uint8(255 * img_sobel / np.max(img_sobel)) # 0..255
	
	# lanes by edge detection
	bin_sobel = binFilter(img_sobel,[(50,255)])
```

#### 5. Binary combination
Just an OR combination of the two images containing all pixels we might consider as lanes.


#### 6. Histogram
To find a starting point to search for lane markings, the bottom half of the binary combined image is used to create a histogram of lane pixels along the x axis.
`helper_functions.py:~234`:
```python
def find_windows(img,WindowsLeft,WindowsRight,minPixels):
    bottom_part_of_image = 0.5
    size_x = img.shape[1]
    size_y = img.shape[0]

    # get histogram for start of lanes (bottom window) (sum over y at each x)
    hist = np.sum(img[int((1-bottom_part_of_image)*size_y):,:], axis = 0)
```

#### 7. Windows & Fitting

Based on the histogram a starting point on each (left/right) half of the image is choosen at the peaks.

#### 7.1 Sliding Windows
For each side the image is split into 10 equaly high widows with a width of 150 px.

Starting at the bottom, each windows start location is (in decending priority) based on the one in the last frame, the one below, or the peak of the histogram.

In this position, the number of pixels in the binary combined image is counted. If the number is high enough (>100) and less than 20% of the windows area (otherwise it might be "all white"), the window is deemed 'valid' and (x-)centered around the weighted mean of the detected pixels and the last center.

By weighting in the windows last position, a great part of jitter is already reduced.

In the example images above, the green squares mark 'valid' windows and the red 'invalid'.

See implementations for `class Window` and `find_windows()` in `helper_functions.py:~131 ~228` for implementation details.

#### 7.2 Polynomal Fitting

Based on each sides windows weighted mean pixels positions, a polinomal of order 2 is fit in `fit_windows()`.

`helper_functions.py:~305`
```python
def fit_windows(windows,xRatio,yRatio,name, ... ):
    pointsX = []
    pointsY = []
    for w in windows:
        if w.valid:
            for px in range(w.get_weight()):
                pointsX.append(w.get_x_mean_total()*xRatio)
                pointsY.append(w.get_y_mean_total()*yRatio)


    as_array = np.array([pointsY,pointsX])
    func_in_y = np.polyfit(as_array[0,:],as_array[1,:],2)
    #...
    return func_in_y 
```

To reduce outliers and jitter, the result is stored in a ringbuffer for 10 frames, the median of these entries is used as the final lane line then (only in video mode).

`find_lanes.py:~118`:
```python
	poly_left = fit_windows(windowsLeft,xRatio,yRatio,...)
	line_l = plot_poly(poly_left,out_size)
	#medians
	line_l = runningMedian(line_l, rolling_median_count, "left")
```

#### 7.3 Numericals

To get numerical values in 'world' coordinates (meters instead of pixels), the fitting is repeated with meter-per-pixel ratios in x and y (`find_lanes.py:~166`). The resulting polynomal is then used to calculate the curvature of the lane lines and the offset of the car to the center of the lane.

Ratios are based on the ones given on the udacity page and altered for the different window size I choose for the perspective.

| dim | m/px |
|:--:|:--|
|x|0.076|
|y|0.03|

Curvature and offset are then averaged.

#### 8. Result

For presentation and video processing the resulting image is transformed back into the old perspective and combined with the original one. curvature and offset are drawn as text into the frame.

The resulting video can be found under [project_video_output.mp4](project_video_output.mp4). Note that the example images in the table above include the valid/invalid windows markers. The video does not.

---

### Discussion

The presented solution works reasonably well (imho) on the project_video, but will get problems especially in difficult lighting conditions. Furthermore the usage of clothoids instead of simple 2nd order polynomals could result in better line fittings. The resolution of meters/pixels is pretty guessed here, so the numerical results shown are likely 'in the right magnitude' but probably not much more. To be useable for real driving, not only the current line should be known to the vehicle, also intersections might be tricky. Averaging over the last few frames results in smoother 'lanes' but also introduces lag, which might be problematic in higher dynamics.