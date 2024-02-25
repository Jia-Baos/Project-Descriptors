#pragma once
#ifndef __SIFT_H__
#define __SIFT_H__

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

/******************************* Defs and macros *****************************/

//像素类型
typedef double pixel_t;

/** default sigma for initial camera smoothing */
#define INIT_SIGMA 0.5

/** default sigma for initial gaussian smoothing */
#define SIGMA 1.6

/** default number of sampled intervals per octave */
#define INTERVALS 3

/** default threshold on keypoint ratio of principle curvatures */
#define RATIO 10

/** maximum steps of keypoint interpolation before failure */
#define MAX_INTERPOLATION_STEPS 5

/** default threshold on keypoint contrast |D(x)| */
#define DXTHRESHOLD 0.03

/** default number of bins in histogram for orientation assignment */
#define ORI_HIST_BINS 36

/** determines gaussian sigma for orientation assignment */
#define ORI_SIGMA_TIMES 1.5

/** determines the radius of the region used in orientation assignment */
#define ORI_WINDOW_RADIUS 3.0 * ORI_SIGMA_TIMES

/** number of passes of orientation histogram smoothing */
#define ORI_SMOOTH_TIMES 2

/** orientation magnitude relative to max that results in new feature */
#define ORI_PEAK_RATIO 0.8

/** length 0f sift feature */
#define FEATURE_ELEMENT_LENGTH 128

/** default number of bins per histogram in descriptor array */
#define DESCR_HIST_BINS 8

/* width of border in which to ignore keypoints */
#define IMG_BORDER 5

/** default width of descriptor histogram array, 4 * 4 = 16 */
#define DESCR_WINDOW_WIDTH 4

/* determines the size of a single descriptor orientation histogram */
#define DESCR_SCALE_ADJUST 3

/* threshold on magnitude of elements of descriptor vector */
#define DESCR_MAG_THR 0.2

/* factor used to convert floating-point descriptor to unsigned char */
#define INT_DESCR_FCTR 512.0

/******************************** Structures *********************************/

struct Keypoint
{
	int octave;										//关键点所在组
	int interval;									//关键点所在层
	double offset_interval;							//调整后的层的增量

	int x;											//x坐标，根据octave和interval可取的层内图像
	int y;											//y坐标，根据octave和interval可取的层内图像

	//scale = sigma0*pow(2.0, o+s/S)
	double scale;									//空间尺度坐标，用于恢复到-1层（第0组第0层）的尺度
	double dx;										//特征点的x坐标，该坐标被恢复到-1层（第0组第0层）的尺度
	double dy;										//特征点的y坐标，该坐标被恢复到-1层（第0组第0层）的尺度

	double offset_x;								//调整后的x方向的增量
	double offset_y;								//调整后的y方向的增量

	double octave_scale;							//关键点所在组的组内尺度
	//高斯金字塔组内各层尺度，不同组的相同层的octave_scale值相同

	double ori;										//关键点方向

	int descr_length;
	double descriptor[FEATURE_ELEMENT_LENGTH];		//128维特征描述子
	double val;										//极值
};

struct MatchPoint
{
	int index_1;									//匹配特征点在图1的下标
	int index_2;									//匹配特征点最小值在图2的下标
};

/**************************** function declaration ****************************/

void ConvertToGray(const Mat& src, Mat& dst);

void DownSample(const Mat& src, Mat& dst);

void UpSample(const Mat& src, Mat& dst);

void GaussianTemplateSmooth(const Mat& src, Mat& dst);

void GaussianSmooth2D(const Mat& src, Mat& dst, double sigma);

void GaussianSmooth(const Mat& src, Mat& dst, double sigma);

void CreateInitSmoothGray(const Mat& src, Mat& dst, double);

void GaussianPyramid(const Mat& src, vector<Mat>& gauss_pyr, int octaves, int intervals, double sigma);

void Sub(const Mat& a, const Mat& b, Mat& c);

void DogPyramid(const vector<Mat>& gauss_pyr, vector<Mat>& dog_pyr, int octaves, int intervals);

void DetectionLocalExtrema(const vector<Mat>& dog_pyr, vector<Keypoint>& extrema, int octaves, int intervals);

void DetectionLocalExtrema_nointerval(const vector<Mat>& dog_pyr, vector<Keypoint>& extrema, int octaves, int intervals);

void Sift(const Mat& src, vector<Keypoint>& features, double sigma = SIGMA, int intervals = INTERVALS);

void Sift_nointerval(const Mat& src, vector<Keypoint>& features, double sigma = SIGMA, int intervals = INTERVALS);

void DrawSiftFeatures(Mat& src, vector<Keypoint>& features);

void DrawKeyPoints(Mat& src, vector<Keypoint>& keypoints);

void DrawKeyPointsRight(Mat& src, vector<Keypoint>& keypoints, cv::Size size);

const char* GetFileName(const char* dir, int i);

void write_pyr(const vector<Mat>& pyr, const char* dir);

void display_pyr(const vector<Mat>& pyr, string dir);

void read_features(vector<Keypoint>& features, const char* file);

void write_features(const vector<Keypoint>& features, const char* file);

void testInverse3D();

#endif // !__SIFT_H__
