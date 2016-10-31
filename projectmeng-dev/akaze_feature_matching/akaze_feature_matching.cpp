#define DEBUG_READ_XML 0

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "../utils.h"

using namespace cv;
using namespace std;

const float inlier_threshold = 20.0f;
const float nn_match_ratio = 0.05f;

void drawEpipolarLines(cv::Mat& image_out, // output image
	cv::Mat& image1, // image 1
	cv::Mat& image2, // image 2
	std::vector<cv::Vec3f> lines1,
	std::vector<cv::Point2f>& points1, // keypoints 1
	std::vector<cv::Point2f>& points2); // keypoints 2

int main(void)
{

	Mat img1 = imread("result_0001.png");
	Mat img2 = imread("result_0002.png");
#if DEBUG_READ_XML
	Mat homography;
	FileStorage fs("H1to3p.xml", FileStorage::READ);
	fs.getFirstTopLevelNode() >> homography;
#endif // DEBUG_READ_XML
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
		double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) + pow(col.at<double>(1) - matched2[i].pt.y, 1));
		if (dist < inlier_threshold)
		{
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);
			good_matches.push_back(DMatch(new_i, new_i, 0));
		}
	}

	//-- Localize the object
	std::vector<Point2f> pts1;
	std::vector<Point2f> pts2;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		pts1.push_back(matched1[good_matches[i].queryIdx].pt);
		pts2.push_back(matched2[good_matches[i].trainIdx].pt);
	}
	
	// Compute fundamental matrix and epipolar lines
	Mat epipolar_11 = img1;
	Mat epipolar_22 = img2;
	Mat epipolar_1122;

	cv::Mat F = cv::findFundamentalMat(cv::Mat(pts1), cv::Mat(pts2), cv::FM_8POINT);

	std::vector<cv::Vec3f> lines1;
	std::vector<cv::Vec3f> lines2;

	cv::computeCorrespondEpilines(pts2, 2, F, lines1);
	cv::computeCorrespondEpilines(pts1, 1, F, lines2);

	drawEpipolarLines(epipolar_11, img1, img2, lines1, pts1, pts2);
	drawEpipolarLines(epipolar_22, img2, img1, lines2, pts2, pts1);

	cv::hconcat(epipolar_11, epipolar_22, epipolar_1122);

	imshow("1122", epipolar_1122);
	//int ii = 0;
	//cout << typeid(img1).name() << endl;

	cvWaitKey(0);
	return 0;
}



void drawEpipolarLines(cv::Mat& image_out, // output image
	cv::Mat& image1, // image 1
	cv::Mat& image2, // image 2
	std::vector<cv::Vec3f> lines1,
	std::vector<cv::Point2f>& points1, // keypoints 1
	std::vector<cv::Point2f>& points2) // keypoints 2)
{
	 // for all epipolar lines
	for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it)
	{
		// Draw the line between first and last column
		cv::line(image_out,
			cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(image2.cols, -((*it)[2] +
			(*it)[0] * image2.cols) / (*it)[1]),
			cv::Scalar(randomFrom(0, 255), randomFrom(0, 255), randomFrom(0, 255)));
	}

	for (std::vector<cv::Point2f>::const_iterator it = points1.begin(); it != points1.end(); it++)
	{
		cv::circle(image1, (*it), 5, cv::Scalar(randomFrom(0,255), randomFrom(0, 255), randomFrom(0, 255)), -1, 8, 0);
	}

}



// ---------------------------------- DO NOT REMOVE ------------------------------------ //

//// Experiment code. Todo: output frontal parallel using the relation between 1st and current
//int main(void)
//{
//
//	Mat img1 = imread("result_0001.png");
//	Mat img2 = imread("result_0002.png");
//#if DEBUG_READ_XML
//	Mat homography;
//	FileStorage fs("H1to3p.xml", FileStorage::READ);
//	fs.getFirstTopLevelNode() >> homography;
//#endif // DEBUG_READ_XML
//	vector<KeyPoint> kpts1, kpts2;
//	Mat desc1, desc2;
//
//	Ptr<AKAZE> akaze = AKAZE::create();
//	akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
//	akaze->detectAndCompute(img2, noArray(), kpts2, desc2);
//
//	BFMatcher matcher(NORM_HAMMING);
//	vector<vector<DMatch>> nn_matches;
//	matcher.knnMatch(desc1, desc2, nn_matches, 2);
//
//	vector<KeyPoint> matched1, matched2, inliers1, inliers2;
//	vector<DMatch> good_matches;
//
//	for (size_t i = 0; i < nn_matches.size(); i++)
//	{
//		DMatch first = nn_matches[i][0];
//		float dist1 = nn_matches[i][0].distance;
//		float dist2 = nn_matches[i][1].distance;
//
//		if (dist1 < nn_match_ratio * dist2)
//		{
//			matched1.push_back(kpts1[first.queryIdx]);
//			matched2.push_back(kpts2[first.trainIdx]);
//		}
//	}
//
//	for (unsigned i = 0; i < matched1.size(); i++)
//	{
//		Mat col = Mat::ones(3, 1, CV_64F);
//		col.at<double>(0) = matched1[i].pt.x;
//		col.at<double>(1) = matched1[i].pt.y;
//#if DEBUG_READ_XML
//		col = homography*col;
//		col /= col.at<double>(2);
//#endif // DEBUG_READ_XML
//
//		double dist = sqrt(	pow(col.at<double>(0) - matched2[i].pt.x, 2) +
//							pow(col.at<double>(1) - matched2[i].pt.y, 2));
//
//		//cout << dist << endl;
//		if (dist < inlier_threshold)
//		{
//			int new_i = static_cast<int>(inliers1.size());
//			inliers1.push_back(matched1[i]);
//			inliers2.push_back(matched2[i]);
//			good_matches.push_back(DMatch(new_i, new_i, 0));
//		}
//	}
//
//	Mat res;
//	drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
//	imwrite("res.png", res);
//
//	//-- Localize the object
//	std::vector<Point2f> obj;
//	std::vector<Point2f> scene;
//
//	for (int i = 0; i < good_matches.size(); i++)
//	{
//		//-- Get the keypoints from the good matches
//		obj.push_back(matched1[good_matches[i].queryIdx].pt);
//		scene.push_back(matched2[good_matches[i].trainIdx].pt);
//	}
//#if 1
//	Mat H = findHomography(obj, scene, CV_RANSAC);
//
//	// Homography from img1 to frontal-parallel view
//	Mat H_10 = (cv::Mat_<double>(3, 3) << 
//		
//		1.257283196548464, 0.0993716665968889, -6.145665573664694e-13,
//		0.2134525863531523, 1.23427760955649, -1.359858406435599e-13,
//		0.0004237178446254059, 0.0004661999569552454, 1);
//	Mat H_20 = H_10 * H;
//	//Mat H_02;
//	//invert(H_20, H_02);
//	//Mat H_12 = H_10 * H_02;
//	//cout << H_1 << endl;
//	Mat warp, frontal, overlap;
//	//homography_warp(img2, H_10 * H, warp);
//	//cout << H_12 << endl;
//	//cout << H << endl;
//	homography_warp(img2, H_20, frontal);
//	homography_warp(img1, H_10, warp);
//
//	cout << H_20 << endl;
//	cout << H_10 << endl;
//
//	//imshow("img1", img1);
//	//imshow("img2", img2);
//	imshow("img2 -> frontal", frontal);
//	imshow("img1 -> frontal", warp);
//#endif
//#if 0
//	double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
//	cout << "A-KAZE Matching Results" << endl;
//	cout << "*******************************" << endl;
//	cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
//	cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
//	cout << "# Matches:                            \t" << matched1.size() << endl;
//	cout << "# Inliers:                            \t" << inliers1.size() << endl;
//	cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
//	cout << endl;
//#endif
//	cvWaitKey(0);
//	return 0;
//}