#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <vector>
#include <cmath>

cv::Mat image1, image2;
cv::Mat tform;
std::vector<cv::Point2f> goodFeatures;
std::vector<cv::Point2f> trackedFeatures;

cv::Mat getWarpImage();
cv::Mat drawFeatureTrace(cv::Mat &src);
cv::Mat drawFeaturePoint(cv::Mat &src);
void matchFeature();
void detectFeature();
void trackFeature(cv::Mat I1, cv::Mat I2);
void onMouse_printCoordinate(int event, int x, int y, int, void*);

int main(int argc, char* argv[])
{
	cv::VideoCapture cap;
	//cap.open("http://192.168.1.1:8080/videofeed?dummy=param.mjpg");
	cap.open(0);
	if (!cap.isOpened())
	{
		std::cout << "fail to open a default camera" << std::endl;
	}
	cv::waitKey(1000);
	cv::Mat current_cap, previous_cap;
	cv::Mat feature_point;
	cv::Mat temp;
	cv::namedWindow("KLT result");
	cv::setMouseCallback("KLT result", onMouse_printCoordinate, 0);
	for (;;)
	{
		cap >> previous_cap;
		cap >> current_cap;
		//cv::resize(previous_cap, previous_cap, cv::Size(640, 480));
		//cv::resize(current_cap, current_cap, cv::Size(640, 480));
		trackFeature(previous_cap, current_cap);
		feature_point = drawFeatureTrace(current_cap);
		//temp = getWarpImage();
		cv::imshow("KLT result", feature_point);
		//cv::imshow("Warp result", temp);
		cv::waitKey(33);
	}
	return EXIT_SUCCESS;
}

cv::Mat getWarpImage()
{
	cv::Mat warp_src;
	tform = cv::findHomography(goodFeatures, trackedFeatures, CV_RANSAC, 1);
	cv::warpPerspective(image2, warp_src, tform, image2.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS + CV_WARP_INVERSE_MAP);
	return warp_src;
}

cv::Mat drawFeaturePoint(cv::Mat &src)
{
	cv::Mat src_copy;
	src.copyTo(src_copy);
	int thinkness = 2;
	int radius = 1;
	std::vector<cv::Point2f>::const_iterator iter = goodFeatures.begin();
	while (iter != goodFeatures.end())
	{
		cv::circle(src_copy, *iter, radius, cv::Scalar(0, 255, 0), thinkness);
		iter++;
	}
	return src_copy;
}

cv::Mat drawFeatureTrace(cv::Mat &src)
{
	cv::Mat src_copy;
	src.copyTo(src_copy);
	std::vector<cv::Point2f>::const_iterator goodFeatures_iter = goodFeatures.begin();
	std::vector<cv::Point2f>::const_iterator trackedFeatures_iter = trackedFeatures.begin();
	while (goodFeatures_iter != goodFeatures.end())
	{
		cv::circle(src_copy, *goodFeatures_iter, 1, cv::Scalar(0, 0, 255), 2, 8, 0);
		cv::circle(src_copy, *trackedFeatures_iter, 1, cv::Scalar(255, 0, 0), 2, 8, 0);
		cv::line(src_copy, *goodFeatures_iter, *trackedFeatures_iter, cv::Scalar(0, 255, 0));
		goodFeatures_iter++;
		trackedFeatures_iter++;
	}
	return src_copy;
}

void matchFeature()
{
	std::vector<uchar> status;
	std::vector<float> error;
	cv::Size winsize = cv::Size(31, 31);
	int maxlevel = 3;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
	double derivlambda = 0.5;
	int flags = 0;
	cv::calcOpticalFlowPyrLK(image1, image2, goodFeatures, trackedFeatures, status, error, winsize, maxlevel, criteria, derivlambda, flags);
}

void detectFeature()
{
	double qualityLevel = 0.01;
	double minDiatance = 10;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;
	int maxCorners = 1000;
	cv::Mat src_gray;
	cv::cvtColor(image1, src_gray, CV_RGB2GRAY);
	cv::goodFeaturesToTrack(src_gray, goodFeatures, maxCorners, qualityLevel, minDiatance, cv::Mat(), blockSize, useHarrisDetector, k);
	//std::cout << goodFeatures.size() << std::endl;
}

void trackFeature(cv::Mat I1, cv::Mat I2)
{
	I1.copyTo(image1);
	I2.copyTo(image2);
	detectFeature();
	matchFeature();
}

void onMouse_printCoordinate(int event, int x, int y, int, void*)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	cv::Point pt = cv::Point(x, y);
	std::cout << "x=" << pt.x << "\t y=" << pt.y << std::endl;// "\t value=" << image1.at<uchar>(y, x) << std::endl;
}