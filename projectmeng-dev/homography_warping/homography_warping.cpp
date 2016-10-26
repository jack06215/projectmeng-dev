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
	Mat image = imread("../../../sample images/WIN_20160527_16_18_50_Pro.jpg");
	Mat image_out;

	String winName = "homography_warping";
	if (image.empty())
	{
		cerr << "Cannot open image" << endl;
		return -1;
	}
	const int max_length = 1000;
	double fx = max_length / static_cast<double>(image.cols);
	double newHeight = static_cast<double>(image.rows) * fx;
	double fy = newHeight / image.rows;
	cv::resize(image, image, cv::Size(0, 0), fx, fy);

	// ------------ warping function (under development) --------------------- //
	rotateMat_t axisRotate;
	axisRotate.f = 4.771474878444084e+02;
	axisRotate.centerX = image.cols / 2;
	axisRotate.centerY = image.rows / 2;
	axisRotate.xrotate = 0.235455723970077;
	axisRotate.yrotate = 0.384897702191884;
	axisRotate.zrotate = -0.0450083774232280;

	rotate_mat_axis(image, image_out, axisRotate);

	namedWindow(winName, WINDOW_NORMAL);
	imshow(winName, image_out);
	cvWaitKey(0);

	return 0;
}
