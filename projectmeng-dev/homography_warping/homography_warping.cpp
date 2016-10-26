#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(void)
{
	Mat image = imread("../../../sample images/WIN_20160527_16_18_50_Pro.jpg");
	String winName = "homography_warping";
	if (image.empty())
	{
		cerr << "Cannot open image" << endl;
		return -1;
	}

	namedWindow(winName, WINDOW_NORMAL);
	imshow(winName, image);
	cvWaitKey(0);

	return 0;
}