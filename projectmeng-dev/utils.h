#include <opencv2/opencv.hpp>


cv::Mat cvMakehgtform(const cv::Mat &input, double xrotate, double yrotate, double zrotate);

cv::Mat cvMakehgtform(const cv::Mat &input, double xrotate, double yrotate, double zrotate)
{
	// Rotation matrices around the X, Y, and Z axis
	cv::Mat RX = (cv::Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, cos(xrotate), -sin(xrotate), 0,
		0, sin(xrotate), cos(xrotate), 0,
		0, 0, 0, 1);
	cv::Mat RY = (cv::Mat_<double>(4, 4) <<
		cos(yrotate), 0, sin(yrotate), 0,
		0, 1, 0, 0,
		-sin(yrotate), 0, cos(yrotate), 0,
		0, 0, 0, 1);
	cv::Mat RZ = (cv::Mat_<double>(4, 4) <<
		cos(zrotate), -sin(zrotate), 0, 0,
		sin(zrotate), cos(zrotate), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);

	// Composed rotation matrix with (RX, RY, RZ)
	cv::Mat R = RX * RY * RZ;
	return R;
}