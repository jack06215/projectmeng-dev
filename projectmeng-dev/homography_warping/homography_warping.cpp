#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <cmath>
#include <iomanip> // setprecision

#include "../utils.h"	// Collection of function I build...

using namespace cv;
using namespace std;

int main(void)
{
	// Load an image
	Mat image = imread_limitedWidth("../../../sample images/result_0001.png", 1000);
	Mat image_out;
	String winName = "homography_warping";
	
	// Rotate the axis of image according to the parameter rotateMat_t
	rotateMat_t axisRotate;
	axisRotate.f = 4.771474878444084e+02;
	axisRotate.centerX = image.cols / 2;
	axisRotate.centerY = image.rows / 2;
	axisRotate.xrotate = 0.188683274687147;
	axisRotate.yrotate = -0.144974545422454;
	axisRotate.zrotate = 0.0900159739991925;

	cv::Mat H;
	rotate_mat_axis(image, image_out, H, axisRotate);

	cout << H << endl;

	// Show the result
	namedWindow(winName, WINDOW_NORMAL);
	imshow(winName, image_out);
	cvWaitKey(0);

	return 0;
}
