#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <vector>


#include <iomanip>
#include <sstream>
#include <string>
#include <windows.h>


#include "../utils.h"

using namespace cv;
using namespace std;

const float inlier_threshold = 2.5f;
const float nn_match_ratio = 0.8f;

void simple_akaze_matching(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2, const float inlier_threshold, const float nn_match_ratio, int save_result = 0);

std::string ExePath();
std::string paddingZeros(int num, int max_zeros);
int numDigits(int number);

//int main(void)
//{
//	cout << paddingZeros(257, 4) << endl;
//	return 0;
//}

#if 1
//Experiment code: H(f'--> i) = H(f' --> i) * H(i --> i+1)
int main(int argc, char* argv[])
{
	// Read the first input image
	//Mat img1 = imread("result_0001.png");
	//std::cout << argv[1] << std::endl;
	//if (img1.empty())
	//{
	//	std::cerr << "Cannot open first frame, exiting the program" << std::endl;
	//	return 3;
	//}

	// Frontal
	Mat H_10 = (cv::Mat_<double>(3, 3) <<
		0.967411297833074, -0.0388960742494462, -419.621181949190,
		-0.0588150626710047, 0.926587591729339, -380.797356154468,
		0.000516160596666294, 0.000783955088496898, 0.540783017453796);

	Mat frontal_1st;

	//homography_warp(img1, H_10, frontal_1st);
	//cv::resize(frontal_1st, frontal_1st, cv::Size(640, 480));
	//cv::resize(frontal_1st, frontal_1st, cv::Size(480, 640));

	cv::String video_name = "output2.avi";

	VideoWriter  video_out(video_name, cv::VideoWriter::fourcc('F', 'M', 'P', '4'), 30, Size(2 * 640, 480));
	//VideoWriter  video_out(video_name, cv::VideoWriter::fourcc('F', 'M', 'P', '4'), 30, Size(2 * 480, 640));

	if (!video_out.isOpened())
	{
		std:cerr << "Colud not open" << video_name << std::endl;
		return 1;
	}
	double t1 = cv::getTickCount();
	for (int i = 1; i < 709; i++)
	{
		std::string inputname = "result_";
		std::string format_1 = ".png";
		std::string save_index = paddingZeros(i+1, 4);
		Mat img2 = imread(inputname + save_index + format_1, -1);

		if (img2.empty())
		{
			std::cerr << "Cannot open image frame, exiting the progam" << std::endl;
			return 2;
		}

		inputname = "result_";
		format_1 = ".png";
		save_index = paddingZeros(i, 4);
		Mat img1 = imread(inputname + save_index + format_1, -1);

		if (img1.empty())
		{
			std::cerr << "Cannot open image frame, exiting the progam" << std::endl;
			return 2;
		}

		std::vector<Point2f> pts1;
		std::vector<Point2f> pts2;

		// Simple AKAZE matching and find homography between them
		simple_akaze_matching(img1, img2, pts1, pts2, inlier_threshold, nn_match_ratio);
		Mat H = findHomography(pts1, pts2, CV_RANSAC);

		//-- Homography from img1 to frontal-parallel view -- //
		// The homogrpahy that warp the 1st frame to frontal-parallel view

		// The homography that warp the 2nd frame to frontal-parallel view, it is the estimated result and it is defined as 
		// the combination of two piece  of homographies
		//		H.inv() : the inverse homography that transforms the 2nd frame to 1st frame.
		//		H_10 : the homography that transforms the 1st to frontal-parallel view.
		Mat H_20 = H_10 * H.inv();
		H_10 = H_20;
		Mat frontal_est, res;


		homography_warp(img2, H_20, frontal_est);
		cv::resize(frontal_est, frontal_est, cv::Size(640, 480));
		cv::resize(img2, img2, cv::Size(640, 480));

		/*cv::resize(frontal_est, frontal_est, cv::Size(480, 640));
		cv::resize(img2, img2, cv::Size(480, 640));*/

		cv::hconcat(img2, frontal_est, res);
		//cout << res.size() << endl;
		double t2 = cv::getTickCount();
		if (t2)
		{
			double tdiff = (t2 - t1) / cv::getTickFrequency();
			double fps = i / tdiff;
			std::cout << "fps: " << fps << std::endl;
		}

		video_out << res;
		waitKey(1);
		//cout << i << " is saved " << endl;
	}
	return 0;
}
#endif


#if 0
//Experiment code: H(f'--> i) = H(f' --> 1) * H(1 --> i)
int main(int argc, char* argv[])
{

	Mat img1 = imread("result_0001.png");
	std::cout << argv[1] << std::endl;
	if (img1.empty())
	{
		std::cerr << "Cannot open first frame, exiting the program" << std::endl;
		return 3;
	}
	Mat H_10 = (cv::Mat_<double>(3, 3) <<
		0.967411297833074, - 0.0388960742494462, - 419.621181949190,
		- 0.0588150626710047,	0.926587591729339, - 380.797356154468,
		0.000516160596666294,	0.000783955088496898,	0.540783017453796);

	Mat frontal_1st;

	homography_warp(img1, H_10, frontal_1st);
	cv::resize(frontal_1st, frontal_1st, cv::Size(640, 480));
	//cv::resize(frontal_1st, frontal_1st, cv::Size(480, 640));

	cv::String video_name = "output.avi";

	VideoWriter  video_out(video_name, cv::VideoWriter::fourcc('F','M','P','4'), 30, Size(2 * 640, 480));
	//VideoWriter  video_out(video_name, cv::VideoWriter::fourcc('F', 'M', 'P', '4'), 30, Size(2 * 480, 640));

	if (!video_out.isOpened())
	{
		std:cerr << "Colud not open" << video_name << std::endl;
		return 1;
	}

	for (int i=2; i < 709; i++)
	{
		std::string inputname = "result_";
		std::string format_1 = ".png";
		std::string save_index = paddingZeros(i,4);
		Mat img2 = imread(inputname + save_index + format_1,-1);

		if (img2.empty())
		{
			std::cerr << "Cannot open image frame, exiting the progam" << std::endl;
			return 2;
		}

		std::vector<Point2f> pts1;
		std::vector<Point2f> pts2;

		// Simple AKAZE matching and find homography between them
		simple_akaze_matching(img1, img2, pts1, pts2, inlier_threshold, nn_match_ratio);
		Mat H = findHomography(pts1, pts2, CV_RANSAC);

		//-- Homography from img1 to frontal-parallel view -- //
		// The homogrpahy that warp the 1st frame to frontal-parallel view

		// The homography that warp the 2nd frame to frontal-parallel view, it is the estimated result and it is defined as 
		// the combination of two piece  of homographies
		//		H.inv() : the inverse homography that transforms the 2nd frame to 1st frame.
		//		H_10 : the homography that transforms the 1st to frontal-parallel view.
		Mat H_20 = H_10 * H.inv();
		Mat frontal_est, res;


		homography_warp(img2, H_20, frontal_est);
		cv::resize(frontal_est, frontal_est, cv::Size(640, 480));
		cv::resize(img2, img2, cv::Size(640, 480));

		/*cv::resize(frontal_est, frontal_est, cv::Size(480, 640));
		cv::resize(img2, img2, cv::Size(480, 640));*/

		cv::hconcat(img2, frontal_est, res);
		//cout << res.size() << endl;


		video_out << res;
		waitKey(1);
		cout << i << " is saved " << endl;
	}
	return 0;
}
#endif
int numDigits(int number)
{
	int digits = 0;
	if (number < 0) digits = 1; // remove this line if '-' counts as a digit
	while (number) 
	{
		number /= 10;
		digits++;
	}
	return digits;
}

std::string paddingZeros(int num, int max_zeros)
{
	std::ostringstream ss;
	ss << std::setw(max_zeros) << std::setfill('0') << num;
	return ss.str();
}

std::string ExePath() 
{
	char buffer[MAX_PATH];
	GetModuleFileNameA(NULL, buffer, MAX_PATH);
	string::size_type pos = string(buffer).find_last_of("\\/");
	return string(buffer).substr(0, pos);
}

void simple_akaze_matching(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2, const float inlier_threshold, const float nn_match_ratio, int save_result)
{
	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;

	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
	akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

	BFMatcher matcher(NORM_HAMMING);
	vector<vector<DMatch>> nn_matches;
	matcher.knnMatch(desc1, desc2, nn_matches, 2);

	vector<KeyPoint> matched1, matched2, inliers1, inliers2;
	vector<DMatch> good_matches;

	for (size_t i = 0; i < nn_matches.size(); i++)
	{
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;

		if (dist1 < nn_match_ratio * dist2)
		{
			matched1.push_back(kpts1[first.queryIdx]);
			matched2.push_back(kpts2[first.trainIdx]);
		}
	}

	for (unsigned i = 0; i < matched1.size(); i++)
	{
		Mat col = Mat::ones(3, 1, CV_64F);
		col.at<double>(0) = matched1[i].pt.x;
		col.at<double>(1) = matched1[i].pt.y;
		double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) + pow(col.at<double>(1) - matched2[i].pt.y, 2));
		//if (dist < inlier_threshold)
		//{
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);
			good_matches.push_back(DMatch(new_i, new_i, 0));
		//}
	}

	if (save_result)
	{
		Mat res;
		drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
		imwrite("res.png", res);
	}

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		pts1.push_back(matched1[good_matches[i].queryIdx].pt);
		pts2.push_back(matched2[good_matches[i].trainIdx].pt);
	}

}