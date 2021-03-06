#define DEBUG_READ_XML 0

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "../utils.h"

using namespace cv;
using namespace std;

const float inlier_threshold = 0.8f;
const float nn_match_ratio = 2.5f;

// ---------------------------------- DO NOT REMOVE ------------------------------------ //
// Experiment code. Todo: output frontal parallel using the relation between 1st and current
int main(void)
{
	// Read input image
	Mat img1 = imread("result_0001.png");
	Mat img2 = imread("result_0002.png");

	// Keypoint and descriptor
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
			//cout << dist1 << " " << dist2 << endl;
			matched1.push_back(kpts1[first.queryIdx]);
			matched2.push_back(kpts2[first.trainIdx]);
		}
		//else
		//{
		//	cout << "Skip" << endl;
		//}
	}

	// cout << kpts1.at(1).pt << endl;

	for (unsigned i = 0; i < matched1.size(); i++)
	{
		Mat col = Mat::ones(3, 1, CV_64F);
		col.at<double>(0) = matched1[i].pt.x;
		col.at<double>(1) = matched1[i].pt.y;

		double dist = sqrt(	pow(col.at<double>(0) - matched2[i].pt.x, 2) +
							pow(col.at<double>(1) - matched2[i].pt.y, 2));

		//std::cout << dist << std::endl;

		if (dist < inlier_threshold)
		{
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);
			good_matches.push_back(DMatch(new_i, new_i, 0));
		}
	}

	Mat res;
	drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
	imshow("res", res);
//
//	//-- Localize the object
//	std::vector<Point2f> pts1;
//	std::vector<Point2f> pts2;
//
//	for (int i = 0; i < good_matches.size(); i++)
//	{
//		//-- Get the keypoints from the good matches
//		pts1.push_back(matched1[good_matches[i].queryIdx].pt);
//		pts2.push_back(matched2[good_matches[i].trainIdx].pt);
//	}
//
//
//#if 1
//	Mat H = findHomography(pts1, pts2, CV_RANSAC);
//
//	//-- Homography from img1 to frontal-parallel view -- //
//	
//	
//	// The homogrpahy that warp the 1st frame to frontal-parallel view
//	Mat H_10 = (cv::Mat_<double>(3, 3) << 
//		
//		1.257283196548464, 0.0993716665968889, -6.145665573664694e-13,
//		0.2134525863531523, 1.23427760955649, -1.359858406435599e-13,
//		0.0004237178446254059, 0.0004661999569552454, 1);
//	
//	// The homography that warp the 2nd frame to frontal-parallel view, it is the estimated result and it is defined as the combination of two piece 
//	// of homographies
//	//		H.inv() : the inverse homography that transforms the 2nd frame to 1st frame.
//	//		H_10 : the homography that transforms the 1st to frontal-parallel view.
//	Mat H_20 = H_10 * H.inv();
//	Mat warp, frontal, overlap;
//
//	homography_warp(img2, H_20, frontal);
//	homography_warp(img1, H_10, warp);
//
//	imshow("img2 -> frontal", frontal);
//	imshow("img1 -> frontal", warp);
//#endif
#if 0
	double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
	cout << "A-KAZE Matching Results" << endl;
	cout << "*******************************" << endl;
	cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
	cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
	cout << "# Matches:                            \t" << matched1.size() << endl;
	cout << "# Inliers:                            \t" << inliers1.size() << endl;
	cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
	cout << endl;
#endif
	cvWaitKey(0);
	return 0;
}