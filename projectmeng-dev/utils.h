#include <opencv2/opencv.hpp>
#include <random>
#include <limits>
#include <utility>
typedef struct
{
	double f;
	double centerX;
	double centerY;
	double xrotate;
	double yrotate;
	double zrotate;
}rotateMat_t;

template <typename T> T randomFrom(const T min, const T max)
{
	static std::random_device rdev;
	static std::default_random_engine re(rdev());
	
	// Create a compile-time conditional 'dist_type' checking on whether the 'value' is a 'real_dist' or 'int_dist'
	typedef typename std::conditional<
		std::is_floating_point<T>::value,
		std::uniform_real_distribution<T>,
		std::uniform_int_distribution<T> > ::type dist_type;

	dist_type uni(min, max);
	return static_cast<T>(uni(re));
}

bool isNumeric(const char* pszInput, int nNumberBase);
bool isFloat(std::string myString);
cv::Mat imread_limitedWidth(cv::String filename, int length_limit, int imread_flag = 1);
cv::Mat cvMakehgtform(double xrotate, double yrotate, double zrotate);
void to_homogeneous(const std::vector< cv::Point2f >& non_homogeneous, std::vector< cv::Point3f >& homogeneous);
void from_homogeneous(const std::vector< cv::Point3f >& homogeneous, std::vector< cv::Point2f >& non_homogeneous);
cv::Rect_<float> get_bounding_box(const std::vector<cv::Point2f>& p);
void homography_warp(const cv::Mat& src, const cv::Mat& H, cv::Mat& dst);
void rotate_mat_axis(const cv::Mat &image, cv::Mat &image_out, rotateMat_t &rotate_parameter);
void rotate_mat_axis(const cv::Mat &image, cv::Mat &image_out, cv::Mat &H, rotateMat_t &rotate_parameter);