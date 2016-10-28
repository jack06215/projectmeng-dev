#include <opencv2/opencv.hpp>
#include <cmath>
#include "utils.h"


bool isNumeric(const char* pszInput, int nNumberBase)
{
	std::string base = "0123456789ABCDEF";
	std::string input = pszInput;

	return (input.find_first_not_of(base.substr(0, nNumberBase)) == std::string::npos);
}

bool isFloat(std::string myString)
{
	std::istringstream iss(myString);
	float f;
	iss >> std::noskipws >> f;	// noskipws considers leading whitespace invalid
								// Check the entire string was consumed and if either failbit or badbit is set
	return iss.eof() && !iss.fail();
}

cv::Mat imread_limitedWidth(cv::String filename, int length_limit, int imread_flag)
{
	// Load an image
	cv::Mat image = imread(filename, imread_flag);

	if (image.empty())
	{
		std::cerr << "Cannot open image" << std::endl;
		return image;
	}

	// Resize if image length is bigger than 10000 px
	if (image.rows > 1000 | image.cols > 1000)
	{
		std::cout << "imread_resize:: Limit the image size to 1000 px in length" << std::endl;
		double fx = length_limit / static_cast<double>(image.cols);
		double newHeight = static_cast<double>(image.rows) * fx;
		double fy = newHeight / image.rows;
		cv::resize(image, image, cv::Size(0, 0), fx, fy);
	}

	return image;
}

cv::Mat cvMakehgtform(double xrotate, double yrotate, double zrotate)
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

// Convert a vector of non-homogeneous 2D points to a vector of homogenehous 2D points.
void to_homogeneous(const std::vector< cv::Point2f >& non_homogeneous, std::vector< cv::Point3f >& homogeneous)
{
	homogeneous.resize(non_homogeneous.size());
	for (size_t i = 0; i < non_homogeneous.size(); i++)
	{
		homogeneous[i].x = non_homogeneous[i].x;
		homogeneous[i].y = non_homogeneous[i].y;
		homogeneous[i].z = 1.0;
	}
}

// Convert a vector of homogeneous 2D points to a vector of non-homogenehous 2D points.
void from_homogeneous(const std::vector< cv::Point3f >& homogeneous, std::vector< cv::Point2f >& non_homogeneous)
{
	non_homogeneous.resize(homogeneous.size());
	for (size_t i = 0; i < non_homogeneous.size(); i++)
	{
		non_homogeneous[i].x = homogeneous[i].x / homogeneous[i].z;
		non_homogeneous[i].y = homogeneous[i].y / homogeneous[i].z;
	}
}

// Transform a vector of 2D non-homogeneous points via an homography.
std::vector<cv::Point2f> transform_via_homography(const std::vector<cv::Point2f>& points, const cv::Matx33f& homography)
{
	// Convert 2D points from Cartesian coordinate to homogeneous coordinate
	std::vector<cv::Point3f> ph;
	to_homogeneous(points, ph);

	// Applied homography
	for (size_t i = 0; i < ph.size(); i++)
	{
		ph[i] = homography*ph[i];
	}

	// Convert (Normalised) the points back to Cartesian coordinate system 
	std::vector<cv::Point2f> r;
	from_homogeneous(ph, r);
	return r;
}

// Find the bounding box of a vector of 2D non-homogeneous points.
cv::Rect_<float> get_bounding_box(const std::vector<cv::Point2f>& p)
{
	cv::Rect_<float> r;
	float x_min = std::min_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.x < rhs.x; })->x;
	float x_max = std::max_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.x < rhs.x; })->x;
	float y_min = std::min_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.y < rhs.y; })->y;
	float y_max = std::max_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.y < rhs.y; })->y;
	return cv::Rect_<float>(x_min, y_min, x_max - x_min, y_max - y_min);
}

// Warp the image src into the image dst through the homography H.
void homography_warp(const cv::Mat& src, const cv::Mat& H, cv::Mat& dst)
{
	// Define four corner points from the input image
	std::vector< cv::Point2f > corners;
	corners.push_back(cv::Point2f(0, 0));
	corners.push_back(cv::Point2f(src.cols, 0));
	corners.push_back(cv::Point2f(0, src.rows));
	corners.push_back(cv::Point2f(src.cols, src.rows));

	// Find the bounding box of the new corner points after applied H
	std::vector< cv::Point2f > projected_corners = transform_via_homography(corners, H);
	cv::Rect_<float> bb = get_bounding_box(projected_corners);

	// Applied translation
	cv::Mat_<double> translation = (cv::Mat_<double>(3, 3) <<
		1, 0, -bb.tl().x,
		0, 1, -bb.tl().y,
		0, 0, 1);

	// Applied resultant rotation + translation warping
	cv::warpPerspective(src, dst, translation * H, bb.size());
}


void rotate_mat_axis(const cv::Mat &image, cv::Mat &image_out, double f, double centerX, double centerY, double xrotate, double yrotate, double zrotate)
{

	// Step 1: Make transformation matrix
	cv::Mat trans4by4 = cvMakehgtform(xrotate, yrotate, 0.0);
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
	double scale_fac = std::abs((std::max(Ref_c_out[1].x, Ref_c_out[2].x) - std::min(Ref_c_out[0].x, Ref_c_out[3].x)) / image.cols); //Based on Length

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

	int maxCols(0), maxRows(0), minCols(0), minRows(0);

	for (int i = 0; i<Ref_c_out.size(); i++)
	{
		if (maxRows < Ref_c_out.at(i).y)
			maxRows = Ref_c_out.at(i).y;

		else if (minRows > Ref_c_out.at(i).y)
			minRows = Ref_c_out.at(i).y;

		if (maxCols < Ref_c_out.at(i).x)
			maxCols = Ref_c_out.at(i).x;

		else if (minCols > Ref_c_out.at(i).x)
			minCols = Ref_c_out.at(i).x;
	}

	// ------------ Warp Z axix ------------------ //
	trans4by4 = cvMakehgtform(0.0f, 0.0f, zrotate);
	cv::Mat R_z = trans4by4(cv::Rect(0, 0, 3, 3));
	H = H * R_z;
	homography_warp(image, H, image_out);
}

void rotate_mat_axis(const cv::Mat &image, cv::Mat &image_out, cv::Mat H, double f, double centerX, double centerY, double xrotate, double yrotate, double zrotate)
{

	// Step 1: Make transformation matrix
	cv::Mat trans4by4 = cvMakehgtform(xrotate, yrotate, 0.0);
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

	H = K_mat * R_mat * K_c * C;

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
	double scale_fac = std::abs((std::max(Ref_c_out[1].x, Ref_c_out[2].x) - std::min(Ref_c_out[0].x, Ref_c_out[3].x)) / image.cols); //Based on Length

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

	int maxCols(0), maxRows(0), minCols(0), minRows(0);

	for (int i = 0; i<Ref_c_out.size(); i++)
	{
		if (maxRows < Ref_c_out.at(i).y)
			maxRows = Ref_c_out.at(i).y;

		else if (minRows > Ref_c_out.at(i).y)
			minRows = Ref_c_out.at(i).y;

		if (maxCols < Ref_c_out.at(i).x)
			maxCols = Ref_c_out.at(i).x;

		else if (minCols > Ref_c_out.at(i).x)
			minCols = Ref_c_out.at(i).x;
	}

	// ------------ Warp Z axix ------------------ //
	trans4by4 = cvMakehgtform(0.0f, 0.0f, zrotate);
	cv::Mat R_z = trans4by4(cv::Rect(0, 0, 3, 3));
	H = H * R_z;
	homography_warp(image, H, image_out);
}

void rotate_mat_axis(const cv::Mat &image, cv::Mat &image_out, rotateMat_t &rotate_parameter)
{

	// Step 1: Make transformation matrix
	cv::Mat trans4by4 = cvMakehgtform(rotate_parameter.xrotate, rotate_parameter.yrotate, 0.0);
	cv::Mat R_mat = trans4by4(cv::Rect(0, 0, 3, 3));

	cv::Mat K_mat = (cv::Mat_<double>(3, 3) <<
		rotate_parameter.f, 0.0, 0.0,
		0.0, rotate_parameter.f, 0.0,
		0.0, 0.0, 1.0);
	cv::Mat K_c = K_mat.clone();
	K_c = K_c.inv();

	cv::Mat C = (cv::Mat_<double>(3, 3) <<
		1, 0, -rotate_parameter.centerX,
		0, 1, -rotate_parameter.centerY,
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
	double scale_fac = std::abs((std::max(Ref_c_out[1].x, Ref_c_out[2].x) - std::min(Ref_c_out[0].x, Ref_c_out[3].x)) / image.cols); //Based on Length

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

	int maxCols(0), maxRows(0), minCols(0), minRows(0);

	for (int i = 0; i<Ref_c_out.size(); i++)
	{
		if (maxRows < Ref_c_out.at(i).y)
			maxRows = Ref_c_out.at(i).y;

		else if (minRows > Ref_c_out.at(i).y)
			minRows = Ref_c_out.at(i).y;

		if (maxCols < Ref_c_out.at(i).x)
			maxCols = Ref_c_out.at(i).x;

		else if (minCols > Ref_c_out.at(i).x)
			minCols = Ref_c_out.at(i).x;
	}

	// ------------ Warp Z axix ------------------ //
	trans4by4 = cvMakehgtform(0.0f, 0.0f, rotate_parameter.zrotate);
	cv::Mat R_z = trans4by4(cv::Rect(0, 0, 3, 3));
	H = H * R_z;
	homography_warp(image, H, image_out);
}

void rotate_mat_axis(const cv::Mat &image, cv::Mat &image_out, cv::Mat &H, rotateMat_t &rotate_parameter)
{

	// Step 1: Make transformation matrix
	cv::Mat trans4by4 = cvMakehgtform(rotate_parameter.xrotate, rotate_parameter.yrotate, 0.0);
	cv::Mat R_mat = trans4by4(cv::Rect(0, 0, 3, 3));

	cv::Mat K_mat = (cv::Mat_<double>(3, 3) <<
		rotate_parameter.f, 0.0, 0.0,
		0.0, rotate_parameter.f, 0.0,
		0.0, 0.0, 1.0);
	cv::Mat K_c = K_mat.clone();
	K_c = K_c.inv();

	cv::Mat C = (cv::Mat_<double>(3, 3) <<
		1, 0, -rotate_parameter.centerX,
		0, 1, -rotate_parameter.centerY,
		0, 0, 1);

	H = K_mat * R_mat * K_c * C;

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
	double scale_fac = std::abs((std::max(Ref_c_out[1].x, Ref_c_out[2].x) - std::min(Ref_c_out[0].x, Ref_c_out[3].x)) / image.cols); //Based on Length

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

	int maxCols(0), maxRows(0), minCols(0), minRows(0);

	for (int i = 0; i<Ref_c_out.size(); i++)
	{
		if (maxRows < Ref_c_out.at(i).y)
			maxRows = Ref_c_out.at(i).y;

		else if (minRows > Ref_c_out.at(i).y)
			minRows = Ref_c_out.at(i).y;

		if (maxCols < Ref_c_out.at(i).x)
			maxCols = Ref_c_out.at(i).x;

		else if (minCols > Ref_c_out.at(i).x)
			minCols = Ref_c_out.at(i).x;
	}

	// ------------ Warp Z axix ------------------ //
	trans4by4 = cvMakehgtform(0.0f, 0.0f, rotate_parameter.zrotate);
	cv::Mat R_z = trans4by4(cv::Rect(0, 0, 3, 3));
	H = H * R_z;
	homography_warp(image, H, image_out);
}