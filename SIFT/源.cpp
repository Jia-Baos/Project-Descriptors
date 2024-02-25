#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "Sift.h"

#ifdef _DEBUG
#pragma comment(lib, "D:/opencv_contrib/build/x64/vc17/lib/opencv_world460d.lib")
#else
#pragma comment(lib, "D:/opencv_contrib/build/x64/vc17/lib/opencv_world460.lib")
#endif // _DEBUG

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat image1 = imread("D:/Code-VS/picture/Demons/lena0.png");
	Mat image2 = imread("D:/Code-VS/picture/Demons/lena1.png");
	Mat image01 = image1.clone();
	Mat image02 = image2.clone();

	vector<Keypoint> features1, features2;
	Sift(image01, features1, SIGMA);
	Sift_nointerval(image02, features2, SIGMA);

	//在图像上绘制特征点
	DrawKeyPoints(image01, features1);
	DrawKeyPoints(image02, features2);

	//绘制特征点主方向
	DrawSiftFeatures(image01, features1);
	DrawSiftFeatures(image02, features2);

	//输出特征点128维特征
	//write_features(features1, "descriptor_1.txt");
	//write_features(features2, "descriptor_2.txt");

	imshow("image01", image01);
	imshow("image02", image02);

	/*cv::imwrite("image1_points.jpg", image01);
	cv::imwrite("image2_points.jpg", image02);*/

	//计算两幅图对应特征点
	cv::Size size;
	size.width = image1.cols + image2.cols;
	size.height = image1.rows > image2.rows ? image1.rows : image2.rows;
	int width = size.width, height = size.height;

	//左右排列图像，为绘制对应特征点做准备
	Mat img_match(size.height, size.width, CV_8UC3);
	for (int i = 0; i < size.height; i++)
	{
		for (int j = 0; j < size.width; j++)
		{
			//加if防止指针超出图像的边界
			if (i < image1.rows && j < image1.cols)
			{
				img_match.data[(i * width + j) * 3] = image1.data[(i * image1.cols + j) * 3];
				img_match.data[(i * width + j) * 3 + 1] = image1.data[(i * image1.cols + j) * 3 + 1];
				img_match.data[(i * width + j) * 3 + 2] = image1.data[(i * image1.cols + j) * 3 + 2];
			}

			//加if防止指针超出图像的边界
			if (i < image2.rows && j >= image1.cols)
			{
				img_match.data[(i * width + j) * 3] = image2.data[(i * image2.cols + j - image1.cols) * 3];
				img_match.data[(i * width + j) * 3 + 1] = image2.data[(i * image2.cols + j - image1.cols) * 3 + 1];
				img_match.data[(i * width + j) * 3 + 2] = image2.data[(i * image2.cols + j - image1.cols) * 3 + 2];
			}
		}
	}

	//因为图像移到了右侧，所以横坐标要修改
	cv::Size size_left;
	size_left.height = image1.rows;
	size_left.width = image1.cols;

	//clone左右排列的图像，绘制特征点
	Mat img_match_keypoints = img_match.clone();
	DrawKeyPoints(img_match_keypoints, features1);
	DrawKeyPointsRight(img_match_keypoints, features2, size_left);
	imshow("img_match_keypoints", img_match_keypoints);

	//存储对应特征点在features中的索引，例如(1, 9)
	vector<MatchPoint> v_matchpoint;
	Mat img_match_line = img_match.clone();

	//用来统计匹配的最近邻和次近邻特征点坐标彼此距离贴近的数量
	int num = 0;

	//特征点匹配
	for (int i = 0; i < features1.size(); i++)
	{
		double odest = 0; //欧氏距离
		//下面三个值可以随机初始化
		int min_index = 0; //最小值的索引
		double min_value = DBL_MAX; //最小值
		int penuntimatemin_index = 0; //次最小值的索引
		double penuntimatemin_value = DBL_MAX; //次最小值
		for (int j = 0; j < features2.size(); j++)
		{
			//计算欧式距离和
			for (int k = 0; k < FEATURE_ELEMENT_LENGTH; k++)
				odest += (features1[i].descriptor[k] - features2[j].descriptor[k]) * (features1[i].descriptor[k] - features2[j].descriptor[k]);

			//odest表示两个特征描绘子矢量的欧氏距离
			odest = sqrt(odest);

			//判断是否是最小
			//如果是最小，将最小赋值到次小，如果不是最小，判断是不是次小
			if (odest < min_value)
			{
				penuntimatemin_value = min_value;
				min_value = odest;
				min_index = j;
			}
			else
			{
				if (odest < penuntimatemin_value)
				{
					penuntimatemin_value = odest;
					penuntimatemin_index = j;
				}
			}
		}

		//std::cout << "min_value / penuntimatemin_value: " << min_value / penuntimatemin_value << std::endl;

		if (0 < min_index < features2.size() && 0 < penuntimatemin_index < features2.size())
		{
			if (abs(features2[min_index].dx - features2[penuntimatemin_index].dx) < 2)
			{
				std::cout << features2[min_index].dx << std::endl;
				std::cout << features2[min_index].scale << std::endl;
				std::cout << features2[penuntimatemin_index].dx << std::endl;
				std::cout << features2[penuntimatemin_index].scale << std::endl;
				num++;
				std::cout << std::endl;
			}
		}
		//判断最小值和次小值的比值是否小于阈值
		//小于阈值加入MatchPoint容器，考虑加上或运算，把位置相近，尺度不同的点也给当作匹配点（测试结果，匹配点太多了）
		if (min_value / penuntimatemin_value < 0.5)
		{
			MatchPoint* matchpoint = new MatchPoint;
			matchpoint->index_1 = i;
			matchpoint->index_2 = min_index;
			v_matchpoint.push_back(*matchpoint);
		}
	}

	//统计匹配的最近邻和次近邻特征点坐标彼此距离贴近的数量
	std::cout << num << std::endl;

	//匹配点的数量
	std::cout << "match number: " << v_matchpoint.size() << std::endl;

	//将两幅图像对应的特征点进行连线
	for (int i = 0; i < v_matchpoint.size(); i++)
	{
		Point point1, point2;
		point1.x = features1[v_matchpoint[i].index_1].dx;
		point1.y = features1[v_matchpoint[i].index_1].dy;
		point2.x = features2[v_matchpoint[i].index_2].dx + image01.cols;
		point2.y = features2[v_matchpoint[i].index_2].dy;
		line(img_match_line, point1, point2, cv::Scalar(0, 255, 0));
	}

	imshow("img_match_line", img_match_line);
	/*imwrite("image_match_line.jpg", img_match_line);*/

	waitKey(0);
	return 0;
}