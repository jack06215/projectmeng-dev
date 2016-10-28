#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>


using namespace cv;
using namespace std;

static double rad2Deg(double rad) { return rad*(180 / M_PI); }//Convert radians to degrees
static double deg2Rad(double deg) { return deg*(M_PI / 180); }//Convert degrees to radians

double    theta;
double    phi;
double    gamma;
double    scale;
double    fovy;

int    g_theta_slider;
int    g_phi_slider;
int    g_gamma_slider;
int theta_slider[720];
int phi_slider[720];
int gamma_slider[720];
int fovy_slider[400];


int    g_scale_slider = 1;
int    g_fovy_slider = 1;

const int	 g_max_slider = 360;

Mat m, disp, warp;
vector<Point2f> corners;


void warpMatrix(Size   sz,
	//double theta,
	//double phi,
	//double gamma,
	//double scale,
	//double fovy,
	Mat&   M,
	vector<Point2f>* corners) {

	double theta = (int)theta_slider[g_theta_slider];
	double phi = (int)phi_slider[g_phi_slider];
	double gamma = (int)gamma_slider[g_gamma_slider];
	double scale = g_scale_slider;
	double fovy = (int)fovy_slider[g_fovy_slider];
	//cout << theta_slider[g_theta_slider] << endl;
	//gamma = -0.0301545504234270;
	//phi = - 0.646843410802792;
	//theta =	0.391526673561161;

	double st = sin(deg2Rad(theta));
	double ct = cos(deg2Rad(theta));
	double sp = sin(deg2Rad(phi));
	double cp = cos(deg2Rad(phi));
	double sg = sin(deg2Rad(gamma));
	double cg = cos(deg2Rad(gamma));

	st = sin(theta);
	ct = cos(theta);
	sp = sin(phi);
	cp = cos(phi);
	sg = sin(gamma);
	cg = cos(gamma);

	double halfFovy = fovy*0.5;
	double d = hypot(sz.width, sz.height);
	double sideLength = scale*d / cos(deg2Rad(halfFovy));
	double h = d / (2.0*sin(deg2Rad(halfFovy)));
	double n = h - (d / 2.0);
	double f = h + (d / 2.0);

	Mat F = Mat(4, 4, CV_64FC1);			//Allocate 4x4 transformation matrix F
	Mat Rtheta = Mat::eye(4, 4, CV_64FC1);	//Allocate 4x4 rotation matrix around Z-axis by theta degrees
	Mat Rphi = Mat::eye(4, 4, CV_64FC1);	//Allocate 4x4 rotation matrix around X-axis by phi degrees
	Mat Rgamma = Mat::eye(4, 4, CV_64FC1);	//Allocate 4x4 rotation matrix around Y-axis by gamma degrees

	Mat T = Mat::eye(4, 4, CV_64FC1);		//Allocate 4x4 translation matrix along Z-axis by -h units
	Mat P = Mat::zeros(4, 4, CV_64FC1);		//Allocate 4x4 projection matrix

	Mat A1 = (Mat_<double>(4, 3) <<
		1, 0, -sz.width / 2,
		0, 1, -sz.height / 2,
		0, 0, 0,
		0, 0, 1);


	//Rtheta
	Rtheta.at<double>(0, 0) = Rtheta.at<double>(1, 1) = ct;
	Rtheta.at<double>(0, 1) = -st; Rtheta.at<double>(1, 0) = st;
	//Rphi
	Rphi.at<double>(1, 1) = Rphi.at<double>(2, 2) = cp;
	Rphi.at<double>(1, 2) = -sp; Rphi.at<double>(2, 1) = sp;
	//Rgamma
	Rgamma.at<double>(0, 0) = Rgamma.at<double>(2, 2) = cg;
	Rgamma.at<double>(0, 2) = sg; Rgamma.at<double>(2, 0) = sg;

	Mat R = Rphi*Rtheta*Rgamma;

	//T
	T.at<double>(2, 3) = -h;
	//P
	P.at<double>(0, 0) = P.at<double>(1, 1) = 1.0 / tan(deg2Rad(halfFovy));
	P.at<double>(2, 2) = -(f + n) / (f - n);
	P.at<double>(2, 3) = -(2.0*f*n) / (f - n);
	P.at<double>(3, 2) = -1.0;

	Mat A2 = (Mat_<double>(3, 4) <<
		f, 0, sz.width / 2, 0,
		0, f, sz.height / 2, 0,
		0, 0, 1, 0);
	Mat A3 = A1 * A2;
	//cout << A3 << endl;
	//cout << P << endl;
	//Compose transformations
	F = P*(T*R);//Matrix-multiply to produce master matrix
				//F = A2*(T*(R*A1));//Matrix-multiply to produce master matrix

				//Transform 4x4 points
	double ptsIn[4 * 3];
	double ptsOut[4 * 3];
	double halfW = sz.width / 2, halfH = sz.height / 2;

	ptsIn[0] = -halfW; ptsIn[1] = halfH;
	ptsIn[3] = halfW; ptsIn[4] = halfH;
	ptsIn[6] = halfW; ptsIn[7] = -halfH;
	ptsIn[9] = -halfW; ptsIn[10] = -halfH;

	ptsIn[2] = ptsIn[5] = ptsIn[8] = ptsIn[11] = 0;//Set Z component to zero for all 4 components

	Mat ptsInMat(1, 4, CV_64FC3, ptsIn);
	Mat ptsOutMat(1, 4, CV_64FC3, ptsOut);

	perspectiveTransform(ptsInMat, ptsOutMat, F);//Transform points

												 //Get 3x3 transform and warp image
	Point2f ptsInPt2f[4];
	Point2f ptsOutPt2f[4];

	for (int i = 0; i<4; i++) {
		Point2f ptIn(ptsIn[i * 3 + 0], ptsIn[i * 3 + 1]);
		Point2f ptOut(ptsOut[i * 3 + 0], ptsOut[i * 3 + 1]);
		ptsInPt2f[i] = ptIn + Point2f(halfW, halfH);
		ptsOutPt2f[i] = (ptOut + Point2f(1, 1))*(sideLength*0.5);
	}

	M = getPerspectiveTransform(ptsInPt2f, ptsOutPt2f);

	//Load corners vector
	if (corners) {
		corners->clear();
		corners->push_back(ptsOutPt2f[0]);	//Push Top Left corner
		corners->push_back(ptsOutPt2f[1]);	//Push Top Right corner
		corners->push_back(ptsOutPt2f[2]);	//Push Bottom Right corner
		corners->push_back(ptsOutPt2f[3]);	//Push Bottom Left corner
	}
}

void warpImage(const Mat &src,
	//double    theta,
	//double    phi,
	//double    gamma,
	//double    scale,
	//double    fovy,
	Mat&      dst,
	Mat&      M,
	vector<Point2f> &corners) {

	double theta = (int)g_theta_slider;
	double phi = (int)g_phi_slider;
	double gamma = (int)g_gamma_slider;
	double scale = g_scale_slider;
	double fovy = g_fovy_slider;

	double halfFovy = fovy*0.5;
	double d = hypot(src.cols, src.rows);
	double sideLength = scale*d / cos(deg2Rad(halfFovy));

	warpMatrix(src.size(), M, &corners);						//Compute warp matrix
	warpPerspective(src, dst, M, Size(sideLength, sideLength));	//Do actual image warp
	circle(dst, Point(corners.at(0)), 3, CV_RGB(255, 0, 0), 3);
	circle(dst, Point(corners.at(1)), 3, CV_RGB(0, 255, 0), 3);
	circle(dst, Point(corners.at(2)), 3, CV_RGB(0, 0, 255), 3);
	circle(dst, Point(corners.at(3)), 3, CV_RGB(255, 255, 0), 3);

	cout << dst.size() << endl;
}

void on_trackbar(int, void*)
{
	warpImage(m, disp, warp, corners);
}


int main(void) {

	VideoCapture cap(0);
	waitKey(1000);
	namedWindow("Disp");
	//m = imread("WIN_20160527_16_21_39_Pro.jpg");
	for (int i = 0; i < sizeof(fovy_slider) / sizeof(fovy_slider[0]); i++)
	{
		fovy_slider[i] = i - 100;
	}
	for (int i = 0; i < sizeof(gamma_slider) / sizeof(gamma_slider[0]); i++)
	{
		theta_slider[i] = i - 360;
		phi_slider[i] = i - 360;
		gamma_slider[i] = i - 360;
		//cout << i - 360 << endl;
	}
	createTrackbar("z-axix", "Disp", &g_theta_slider, 719, &on_trackbar);
	createTrackbar("y-axix", "Disp", &g_phi_slider, 719, &on_trackbar);
	createTrackbar("x-axix", "Disp", &g_gamma_slider, 719, &on_trackbar);
	createTrackbar("fovy", "Disp", &g_fovy_slider, 200, &on_trackbar);
	while (cap.isOpened()) {
		cap >> m;
		warpImage(m, disp, warp, corners);
		imshow("Disp2", disp);
		waitKey(33);
	}
	return 0;
}

//// -------------------- old Attempt ---------------------
//Mat frame;
//
//int alpha_int = -100;
//int dist_int;
//int f_int;
//
//double w;
//double h;
//double alpha;
//double dist;
//double f;
//Mat frame1;
//
//void redraw(Mat& frame1) {
//
//	alpha = (double)alpha_int / 1000.;
//	//dist = 1./(dist_int+1);
//	//dist = dist_int+1;
//	dist = dist_int - 50;
//	f = f_int + 1;
//
//	cout << "alpha = " << alpha << endl;
//	//cout << "dist = " << dist << endl;
//	//cout << "f = " << f << endl;
//
//	// Projection 2D -> 3D matrix
//	Mat A1 = (Mat_<double>(4, 3) <<
//		1, 0, -w / 2,
//		0, 1, -h / 2,
//		0, 0, 1,
//		0, 0, 1);
//
//	// Rotation matrices around the X axis
//	Mat R = (Mat_<double>(4, 4) <<
//		1, 0, 0, 0,
//		0, cos(deg2Rad(alpha)), -sin(deg2Rad(alpha)), 0,
//		0, sin(deg2Rad(alpha)), cos(deg2Rad(alpha)), 0,
//		0, 0, 0, 1);
//
//	// Translation matrix on the Z axis 
//	Mat T = (Mat_<double>(4, 4) <<
//		1, 0, 0, 0,
//		0, 1, 0, 0,
//		0, 0, 1, dist,
//		0, 0, 0, 1);
//
//	// Camera Intrisecs matrix 3D -> 2D
//	Mat A2 = (Mat_<double>(3, 4) <<
//		f, 0, w / 2, 0,
//		0, f, h / 2, 0,
//		0, 0, 1, 0);
//
//	Mat m = A2 * (T * (R * A1));
//
//	//cout << "R=" << endl << R << endl;
//	//cout << "A1=" << endl << A1 << endl;
//	//cout << "R*A1=" << endl << (R*A1) << endl;
//	//cout << "T=" << endl << T << endl;
//	//cout << "T * (R * A1)=" << endl << (T * (R * A1)) << endl;
//	//cout << "A2=" << endl << A2 << endl;
//	//cout << "A2 * (T * (R * A1))=" << endl << (A2 * (T * (R * A1))) << endl;
//	//cout << "m=" << endl << m << endl;
//
//
//
//	warpPerspective(frame, frame1, m, frame.size(), INTER_CUBIC | WARP_INVERSE_MAP);
//
//	//imshow("Frame", frame);
//	//imshow("Frame1", frame1);
//}
//
//void callback(int, void*) {
//	redraw(frame1);
//}
//
//void main() {
//
//	VideoCapture cap(0);
//	waitKey(1000);
//	namedWindow("Frame");
//	dist_int = 50;
//	createTrackbar("alpha", "Frame", &alpha_int, 360, &callback);
//	while (cap.isOpened()) {
//		cap >> frame;
//		w = frame.size().width;
//		h = frame.size().height;
//		redraw(frame1);
//		imshow("Frame", frame);
//		imshow("Frame1", frame1);
//		waitKey(33);
//	}
//
//	waitKey(-1);
//}