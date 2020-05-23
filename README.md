## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![alt text][image6]

In this project, the goal is to write a software pipeline to identify the lane boundaries in a video.

**Note: For the writeup refer either to the writeup file `WRITEUP.md` or see the sections below.**

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw input images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image1a]: ./camera_cal/calibration1.jpg "Distorted calib image"
[image1b]: ./output_images/undistorted_calibration1.jpg "Undistorted calib image"
[image2]: ./output_images/undistorted_test5.jpg "Undistorted example image"
[image3]: ./output_images/binary_7.png "Binary Example"
[image4]: ./output_images/transformed_straight_lines1.jpg "Warp Example"
[image5]: ./output_images/windowing_example.png "Windowing and polynomial fit example"
[image6]: ./output_images/result_test4.jpg "Overall output with projection"
[video1]: ./project_video.mp4 "Video"

## General notes

All code is contained in the iPython Notebook [P2.ipynb](./P2.ipynb) and is throughout commented. All the respective segments of the pipeline are highlighted in this file and speak for themselves. Hence, there is no specific necessity to provide line numbers.

All significant processing steps are encapsulated in functions and are subsequently called by the main pipeline function.

## Rubric Points

[See rubric online](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md)

Here, I consider the rubric points individually and describe how each point is addressed in the implementation.  

---

### Writeup


##### Provide a Writeup / README that includes all the rubric points and how each one is addressed. 
 
This is provided - it is this document ;-)

### Camera Calibration

##### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

For calibrating the camera, I have defined the function `calibrate_camera`. It takes as input: the number of rows and columns of the chessboard intersections, the folder containing the camera images, the output folder for calibrated images and a parameter for activating/deactivating plots.

Generically, I started by defining "object points" and "image points". Object points represent the position of our chessboard intersections in real world - thus, it is represented as (x, y, z) coordinates. To use this as a defined calibration medium, I assume an ideal chessboard with the z coordinate set to 0 (perfect plane). This is applied for all calibration images as we use an identical chessboard printout in all images. Correspondingly, image points represent the pixel positions in 2D (x, y) of the respective intersections. These two arrays of points are calculated for each image and appended to two respective arrays containing all the object and image points.

These object and imagepoints are then used to calculate the camera calibration and distortion coefficients using `cv2.calibrateCamera()`.

Images are then undistorted using the `cv2.undistort()`(encapsulated in `undistort()`).

![alt text][image1a]
*Fig 1a: Distorted calibration image*

![alt text][image1b]
*Fig 1b: Undistorted image*

### Pipeline (single images)

##### Provide an example of a distortion-corrected image.

To demonstrate this step, a code segment was temporarily included that outputs the result of `cv2.undistort()` (encapsulated in `undistort()`).
An example of an undistorted image is provided below.

![alt text][image2]
*Fig 2: Undistorted test image*


##### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code is in the section "Create threshold binary image" and is encapsulated within the function `binary_threshold()`. The function takes the image to be proecessed and the respective thresholds for "thresholding" specific color channels and gradient planes. 

To create the binary image, I combined a distinct color channel and gradient: (a) saturation from the HLS color space and (b) grayscale gradient along the x-axis (see code in section "Create threshold binary image"). For (a) I have converted the image into HLS color space and isolated the saturation channel - subsequently applied a threshold to filter the relevant pixels in the image:

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1


For (b) I have employed the sobel function `cv2.Sobel()` to calculate the gradient along the x-axis for identifying "sharp" edges in vertical direction - assuming these would identify lane lines. Subsequently a thresholding is applied, which delivers lane line pixels in the image:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sobelx_binary[(scaled_sobel >= x_thresh_min) & \
        (scaled_sobel <= x_thresh_max)] = 1
    
    
Eventually these two components are combined using a bitwise "OR" operator to generate a combined binary image.

    combined_binary[(s_binary == 1) | (sobelx_binary == 1)] = 1

The below image highlights the two components with (a) represented as green and (b)represented as blue (only for illustration purposes).

![alt text][image3]
*Fig 2: Undistorted, stacked binary image (S-channel in green and grayscale sobel along x-axis in blue) and combined binary image*

Further experimentation with H and L-channels did not yield significant improvements.


#### Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code is in the section "Perform perspective transform" and is encapsulated within the two functions (a) `transform_matrix()` for calculating the transform matrix and (b) `warp` for warping the image using the calculated transform matrix. The function takes a binary image and parameters for setting the source points for the transform.

Within `transform_matrix()` I define two arrays holding source and destination points. The (a) destination points  are defined along the height, width and a width-rescaling parameter of the image. As I want to create a top down image that maps the perspective-wise distorted image points (looking like a trapezoid) it is a rectangle.

For calculating (b) the source points, I assume, based on the undistorted camera image and the camera position, that the lane lines on the left and right fill a trapezoid area on the image plane. For this I subjectively optimized a set of trapezoid parameters (height as well as width on the top and bottom of the trapezoid) to define a geometrical set of vertices as source points:


    source = np.float32([[imshape[1] * (1-trans_bot_width), imshape[0]],\
       [imshape[1] * (1-trans_top_width), imshape[0] * (1-trans_height)],\
       [imshape[1] * trans_top_width, imshape[0] * (1-trans_height)],\
       [imshape[1]* trans_bot_width, imshape[0]]])
       

When transforming an image with straight lane lines, the transformed image also shows straight lane lines, thus verifies correctness of the transform:

![alt text][image4]
*Fig 3: Warped image of example 'straight_lines1'*

##### Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To detect lane-line pixels I defined a function `find_lane_pixels()` that takes a warped binary image and windowing parameters as argument.

The function initially detects the bottom of the lane lines by calculating a histogram of the bottom half of the image. Thereby the respective maximum peaks represent the bottom position for the right and left lane lines:

    hist = histogram_half(img) # histogram of bottom half of 'img'
    midpoint = np.uint(hist.shape[0]//2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint

With the respective lane base positions we start iterating vertically by detecting lane line pixels that lie within a defined window (function arguments define number of vertical windows and horizontal margin of each window). Thereby, I set a somewhat arbitrary minimum requirement of 50 pixels to be identified in order to move the window to the next vertical position (along y-axis) based on the mean of the pixel x-values. Otherwise, I assume the lane line pixels were not well detected and therefore the window is moved vertically, without adjustment along the x-axis.

All identified lane-line pixels are stored in arrays - which are then passed to the function `fit_polynomial()` incl. the binary image.

Hereby, I assume a second-order polynomial to capture lane curvature and fit it using `np.polyfit()`:
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

Then, the polynomial is plotted on the warped binary image using `cv2.polylines()`.

The corredponding code for both `find_lane_pixels()` and `fit_polynomial()` is included in the section "Identify lane line pixels and fit polynomials" in the notebook.

![alt text][image5]
*Fig. 4: Windowing along detected lane pixels incl. polynomial fit and histogram for detecting lane starting points*

##### Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature as well as the lane position of the vehicle are calculated in the section "Calculate radius of curvature and vehicle offset".

For this, I defined a function `curvature()` that takes left and right lane-line pixel positions (x, y), polynomial coefficients from previous fit, and a scale-ratio.

Important here is, that I want to calculate the radius of curvature in real world meters - and not as in pixels. Therefore, I use the scale-ratio (meters per pixel) to scale each y and x coordinate of each point to yield meters. This is then used to calculate adjusted coefficients:

    left_fit = np.polyfit(lefty * ym_per_pixel, leftx * xm_per_pixel, 2)
    right_fit = np.polyfit(righty * ym_per_pixel, rightx * xm_per_pixel, 2)

These coefficients are then used to calculate the radius of curvature at the bottom of the image (can be looked up in mathematics formulary):
    
    radius_left = (1 + (2 * left_fit[0] * ybottom_scaled \
        + left_fit[1]) **2) **1.5 / np.absolute(2 * left_fit[0])
    radius_right = (1 + (2 * right_fit[0] * ybottom_scaled \
        + right_fit[1]) **2) **1.5 / np.absolute(2 * right_fit[0])

Additionally, the vehicle offset from the lane midpoint is calculated by avergaing the positions of the left and right lane at the bottom of the image and calculating the difference vs. the midpoit of the image along the x-axis:

    offset = (midpoint - np.mean([leftx_base, rightx_base])) * xm_per_pixel

##### Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is implemented in the function `lane_projection()` in the section "Output lane boundaries on original image" (this is done on the undistorted orig image).

To project the identified lane lines and valid lane area, I first start with drawing the lane lines on the warped image space (emtpy array with identical shape).
    
    cv2.polylines(img_lines, [pts_left], False, (0, 0, 255), 40) #blue
    cv2.polylines(img_lines, [pts_right], False, (255, 0, 0), 40) #red
    
Then, I fill the area inbetween with a green color to denote the identified lane area:

    lane = np.array([pts_left, pts_right[::-1]]).reshape(-1,1,2)
    cv2.fillPoly(img_lines, [lane], color=(0,100,0))
    
Then the image is warped back to the original perspective using the inverse transform matrix from before and blend the output with the original undistorted image:
    
    img_projectedlines = warp(img_lines, Minv)
    output_img = cv2.addWeighted(img, 1.0, img_projectedlines, 1.0, 0)

Eventually, I add text to the image by using `cv2.putText()` and show both mean curvature radius and vehicle offset vs. lane midpoint:

    cv2.putText(output_img, text_radius, (20,50),\
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(output_img, text_offset, (20,80),\
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

![alt text][image6]
*Fig 5: Left and right lane boundaries and lane area projected on undist. orig. image*

---

### Pipeline (video)

##### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The processed video of `project_video.mp4` can be found under this  [link](./output_videos/processed_project_video.mp4)

---

### Discussion

##### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

#### Retrospective
Implementing this pipeline required a set of various techniques (OpenCV library is very helpful here):
* Calibrating the camera
* Undistorting images
* Warping images into top-down perspective
* Identifying lane line pixels using color spaces and gradients, as well as combining these into a single binary image
* Fitting a polynomial to the identified lane line pixels
* Calculating curvature radius and vehicle offset
* Using transform to project lane lines on the original image

I have put effort in encapsulating the specific processing steps into their own functions to increase reuasability and readibility of the pipeline. This created a much simpler structure to read and find specific parts of the code. However, it made it significantly more complex to implement further improvements, as arguments need to be adopted in order to pass new data into the function - which is necesarry for e.g., keeping track of the lane line telemetry in a `class Line()`. For this, all almost all functions need to provide data and hence need to return the relevant information, which then need to be passed to the `Line` objects.

Hence, it might be a better idea not to encapsulate "too much" during prototyping in order to leave enough space for figuring out what is actually needed and to experiment with the code. Otherwise, "code logistics" grow and create obstructive overhead.

#### Robustness of detection
The pipeline is somewhat robust throughout the image frames in the project video and correctly highlights the lane lines.

However, some frames cause the detection to behave slighlty "wobbly" and make it necessary to make lane detection more robust to shadows and other influences such as vehicles crossing the lanes.

To improve this, it might be necessary to further fine-tune binary image generation by considering further color spaces to extract relevant lane line information from the image. This could be, for instance, the HSV color space and its respective color channels.

Another approach to improve detection robustness could be to include smoothing in combination with specific color channels and combining it with sobel gradient on these color channels.

While the pipeline yielded good results on the project video, the challenge video caused significant problems. The algorithm mistakenly took asphalt repair lines as lane line and abruptly would have caused the vehicle to leave the actual lane.

This requires for a more fine-grained detection approach and sanity check for disallowing unreasonably abrupt shifts of the lane line along the x-axis. Furthermore, the polynomial fit should be restricted so that it can only yield coefficients that are realistic in nature.

Concluding, the following approaches might help to increase robustness:
* Sanity check: check for realistic curvatures, check for a realistic distance between left and right lane lines, check for significant parallelism of both lane lines
* Outlier detection: it might be helpful to identify invalid detections (based on sanity checks) and to disregard these results if only present for a limited time (e.g., 3-5 frames) and to continue again as soon as the results are plausible again
* Smoothing: smoothing the result over multiple frames can help to reduce "jitter" and enhance accuracy by taking the information from multiple neighbouring image frames

#### Computational efficiency
The pipeline is so far able to calculate ~10 frames per second on the local machine. For real-time application this might be too slow, as ordinary cameras record at roughly 30 frames per second. This would mean that an increase in computational speed of at least the factor 3 is required to be capable of real-time detection.

One approach for improving speed might be to reduce the occurance of computationally expensive parts of the pipeline. Specifically, the windowing approach seems to be a rather slow approach that iterates 18 times through an image before the detected lane pixel values can be stored.

This could be significantly reduced by using a look-ahead filter that assumes similarity between the current and the upcoming frame regarding the position of the lane-line pixels. One could implement code that only looks for lane pixels around the currently determined polynomial with a specific margin instead of going through the entire windowing process and initially calculating the histogram. This could yield significant speed benefits and might also help increasing robustness vs. outliers (e.g., vehicles identified as lane pixels).

Another approach for improving computational efficiency might be to process independent code parts in parallel by implementing threads. Currently with this 4-core processor the python kernel only utilizes ~25% of cpu time. This could be significantly increased by putting independent code segments in separate threads. For prototyping purposes this might create significant complexity, however seems necessary for keeping hardware requirements low for production evironments (such as in real cars).