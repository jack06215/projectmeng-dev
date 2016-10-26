#include <opencv2/opencv.hpp>


typedef struct
{
	double f;
	double centerX;
	double centerY;
	double xrotate;
	double yrotate;
	double zrotate;
}rotateMat_t;

cv::Mat cvMakehgtform(double xrotate, double yrotate, double zrotate);
void to_homogeneous(const std::vector< cv::Point2f >& non_homogeneous, std::vector< cv::Point3f >& homogeneous);
void from_homogeneous(const std::vector< cv::Point3f >& homogeneous, std::vector< cv::Point2f >& non_homogeneous);
cv::Rect_<float> get_bounding_box(const std::vector<cv::Point2f>& p);
void homography_warp(const cv::Mat& src, const cv::Mat& H, cv::Mat& dst);
void rotate_mat_axis(const cv::Mat &image, cv::Mat &image_out, double f, double centerX, double centerY, double xrotate, double yrotate, double zrotate);
void rotate_mat_axis(const cv::Mat &image, cv::Mat &image_out, rotateMat_t &rotate_parameter);