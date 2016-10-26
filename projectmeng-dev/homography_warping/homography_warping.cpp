#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <cmath>
#include <iomanip> // setprecision

#include "../utils.h"

using namespace cv;
using namespace std;

cv::Mat cvMakehgtform(const cv::Mat &input, double xrotate, double yrotate, double zrotate);

int main(void)
{
	Mat image = imread("../../../sample images/WIN_20160527_16_18_50_Pro.jpg");
	Mat image_out;
	//Mat H = (Mat_<double>(3, 3) <<
	//					1, 0, 0,
	//					0, 1, 0,
	//					0, 0, 1);

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
	bool talk = true;
	double f = 4.771474878444084e+02;
	double centerX = image.cols / 2;
	double centerY = image.rows / 2;

	// Step 1: Make transformation matrix
	cv::Mat trans4by4 = cvMakehgtform(image, 0.235455723970077, 0.384897702191884, 0.0);
	cv::Mat R_mat = trans4by4(cv::Rect(0, 0, 3, 3));

	cv::Mat K_mat = (cv::Mat_<double>(3, 3) << 
									f, 0.0, 0.0, 
									0.0, f, 0.0, 
									0.0, 0.0, 1.0);
	cv::Mat K_c = K_mat.clone();
	K_c = K_c.inv();

	cv::Mat C = (cv::Mat_<double>(3, 3) << 
								1, 0, -centerX, 
								0, 1, -centerY, 
								0, 0, 1);

	cv::Mat H = K_mat * R_mat * K_c * C;

	// Step 2: Calclating Resultant Translation and Scale
	std::vector<cv::Point2f> Ref_c;
	std::vector<cv::Point2f> Ref_c_out;
	Ref_c.resize(4);
	Ref_c_out.resize(4);	

	Ref_c[0].x = 0;		Ref_c[0].y = 0; 									// top-left
	Ref_c[1].x = double(image.cols);	Ref_c[1].y = 0;						// top-right
	Ref_c[2].x = double(image.cols);	Ref_c[2].y = double(image.rows);	// bottom-right
	Ref_c[3].x = 0;		Ref_c[3].y = double(image.rows);					// bottom-left

	cv::perspectiveTransform(Ref_c, Ref_c_out, H);

	//Scalling:
	double scale_fac = abs((max(Ref_c_out[1].x, Ref_c_out[2].x) - min(Ref_c_out[0].x, Ref_c_out[3].x)) / image.cols); //Based on Length
	


	// Re-scale 4 corner points by the scale_fac
	Ref_c_out[0].x = Ref_c_out[0].x / scale_fac;
	Ref_c_out[0].y = Ref_c_out[0].y / scale_fac;
	Ref_c_out[1].x = Ref_c_out[1].x / scale_fac;
	Ref_c_out[1].y = Ref_c_out[1].y / scale_fac;
	Ref_c_out[2].x = Ref_c_out[2].x / scale_fac;
	Ref_c_out[2].y = Ref_c_out[2].y / scale_fac;
	Ref_c_out[3].x = Ref_c_out[3].x / scale_fac;
	Ref_c_out[3].y = Ref_c_out[3].y / scale_fac;

	Ref_c_out[1].x = Ref_c_out[1].x - Ref_c_out[0].x;
	Ref_c_out[1].y = Ref_c_out[1].y - Ref_c_out[0].y;
	Ref_c_out[2].x = Ref_c_out[2].x - Ref_c_out[0].x;
	Ref_c_out[2].y = Ref_c_out[2].y - Ref_c_out[0].y;
	Ref_c_out[3].x = Ref_c_out[3].x - Ref_c_out[0].x;
	Ref_c_out[3].y = Ref_c_out[3].y - Ref_c_out[0].y;
	Ref_c_out[0].x = Ref_c_out[0].x - Ref_c_out[0].x;
	Ref_c_out[0].y = Ref_c_out[0].y - Ref_c_out[0].y;

	//For the translated/scalled image
	H = getPerspectiveTransform(Ref_c, Ref_c_out); 

	//cout << R_mat << endl;
	//cout << C << endl;
	//cout << scale_fac << endl;
	cout << H << endl;

	warpPerspective(image, image_out, H, image.size(), INTER_LANCZOS4);
	namedWindow(winName, WINDOW_NORMAL);
	imshow(winName, image_out);
	cvWaitKey(0);

	return 0;
}

