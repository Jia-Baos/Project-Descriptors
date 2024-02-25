#include "Sift.h"

//使得printf可用
#define _CRT_SECURE_NO_DEPRECATE

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "sift.h"
using namespace std;

/*
  将图像由uchar型RGB三通道彩色图像转化为double型一通道灰度图像
  @param src 输入图像
  @param dst 输出图像
*/
void ConvertToGray(const Mat& src, Mat& dst)
{
	Size size = src.size();
	if (dst.empty())
		dst.create(size, CV_64F);

	//创建指针指向src的首地址
	uchar* srcData = src.data;
	//创建指针指向dst的首地址
	pixel_t* dstData = (pixel_t*)dst.data;

	//step[0] 代表一行的数据大小(Byte)
	//step[1] 代表一个元素的数据大小(Byte)
	//dstData[0] 代表每一个像素所占的字节数
	//dstStep 每行的像素个数
	//这样访问可把图像当作一维数组处理，对于卷积等运算变得更加方便
	int dstStep = dst.step / sizeof(dstData[0]);

	for (int j = 0; j < src.cols; j++)
	{
		for (int i = 0; i < src.rows; i++)
		{
			double b = *(srcData + src.step * i + src.channels() * j + 0) / 255.0;
			double g = *(srcData + src.step * i + src.channels() * j + 1) / 255.0;
			double r = *(srcData + src.step * i + src.channels() * j + 2) / 255.0;

			*((dstData + dstStep * i + dst.channels() * j)) = b * 0.114 + g * 0.587 + r * 0.299;
		}
	}
}

/*
  对double型一通道灰度图像进行二元下采样
  @param src 输入图像
  @param dst 输出图像
*/
void DownSample(const Mat& src, Mat& dst)
{
	if (src.channels() != 1)
		return;

	if (src.cols <= 1 || src.rows <= 1)
	{
		src.copyTo(dst);
		return;
	}

	dst.create((int)(src.rows / 2), (int)(src.cols / 2), src.type());

	//创建指针指向src的首地址
	pixel_t* srcData = (pixel_t*)src.data;
	//创建指针指向dst的首地址
	pixel_t* dstData = (pixel_t*)dst.data;

	//srcStep 每行的像素个数
	int srcStep = src.step / sizeof(srcData[0]);
	//dstStep 每行的像素个数
	int dstStep = dst.step / sizeof(dstData[0]);

	int m = 0, n = 0;
	for (int j = 0; j < src.cols; j += 2, n++)
	{
		m = 0;
		for (int i = 0; i < src.rows; i += 2, m++)
		{
			//对原图像进行二元下采样，结果传给sample
			pixel_t sample = *(srcData + srcStep * i + src.channels() * j);

			//防止当图像长宽不一致时，长宽为奇数时，m,n越界
			if (m < dst.rows && n < dst.cols)
			{
				//将二元下采样的结果传给输出dst
				*(dstData + dstStep * m + dst.channels() * n) = sample;
			}
		}
	}
}

/*
  对double型一通道灰度图像进行二元上采样（插值）
  @param src 输入图像
  @param dst 输出图像
*/
void UpSample(const Mat& src, Mat& dst)
{
	if (src.channels() != 1)
		return;
	dst.create(src.rows * 2, src.cols * 2, src.type());

	//创建指针指向src的首地址
	pixel_t* srcData = (pixel_t*)src.data;
	//创建指针指向dst的首地址
	pixel_t* dstData = (pixel_t*)dst.data;

	//srcStep 每行的像素个数
	int srcStep = src.step / sizeof(srcData[0]);
	//dstStep 每行的像素个数
	int dstStep = dst.step / sizeof(dstData[0]);

	int m = 0, n = 0;
	for (int j = 0; j < src.cols - 1; j++, n += 2)
	{
		m = 0;
		for (int i = 0; i < src.rows - 1; i++, m += 2)
		{
			//先将原图像的像素值copy到dst的index为偶数的位置上
			double sample = *(srcData + srcStep * i + src.channels() * j);
			*(dstData + dstStep * m + dst.channels() * n) = sample;

			//行插值
			double rs = *(srcData + srcStep * (i)+src.channels() * j) + (*(srcData + srcStep * (i + 1) + src.channels() * j));
			*(dstData + dstStep * (m + 1) + dst.channels() * n) = rs / 2;

			//列插值
			double cs = *(srcData + srcStep * i + src.channels() * (j)) + (*(srcData + srcStep * i + src.channels() * (j + 1)));
			*(dstData + dstStep * m + dst.channels() * (n + 1)) = cs / 2;

			//对角位置插值
			double center = (*(srcData + srcStep * (i + 1) + src.channels() * j))
				+ (*(srcData + srcStep * i + src.channels() * j))
				+ (*(srcData + srcStep * (i + 1) + src.channels() * (j + 1)))
				+ (*(srcData + srcStep * i + src.channels() * (j + 1)));

			*(dstData + dstStep * (m + 1) + dst.channels() * (n + 1)) = center / 4;
		}
	}

	if (dst.rows < 3 || dst.cols < 3)
		return;

	//最后两列
	for (int k = dst.rows - 1; k >= 0; k--)
	{
		*(dstData + dstStep * (k)+dst.channels() * (dst.cols - 2)) = *(dstData + dstStep * (k)+dst.channels() * (dst.cols - 3));
		*(dstData + dstStep * (k)+dst.channels() * (dst.cols - 1)) = *(dstData + dstStep * (k)+dst.channels() * (dst.cols - 3));
	}
	//最后两行
	for (int k = dst.cols - 1; k >= 0; k--)
	{
		*(dstData + dstStep * (dst.rows - 2) + dst.channels() * (k)) = *(dstData + dstStep * (dst.rows - 3) + dst.channels() * (k));
		*(dstData + dstStep * (dst.rows - 1) + dst.channels() * (k)) = *(dstData + dstStep * (dst.rows - 3) + dst.channels() * (k));
	}
}

/*
  利用7*7高斯卷积核对uchar型图像进行平滑
  @param src 输入图像
  @param dst 输出图像
*/
void GaussianTemplateSmooth(const Mat& src, Mat& dst)
{
	//高斯模板(7*7)，sigma = 0.84089642，归一化后得到
	static const double gaussianTemplate[7][7] =
	{
		{0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067},
		{0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292},
		{0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117},
		{0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771},
		{0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117},
		{0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292},
		{0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067}
	};

	dst.create(src.size(), src.type());

	//创建指针指向src的首地址
	uchar* srcData = src.data;
	//创建指针指向dst的首地址
	uchar* dstData = dst.data;

	for (int j = 0; j < src.cols - 7; j++)
	{
		for (int i = 0; i < src.rows - 7; i++)
		{
			double acc = 0;
			double accb = 0, accg = 0, accr = 0;
			for (int m = 0; m < 7; m++)
			{
				for (int n = 0; n < 7; n++)
				{
					if (src.channels() == 1)
						acc += *(srcData + src.step * (i + n) + src.channels() * (j + m)) * gaussianTemplate[m][n];
					else
					{
						accb += *(srcData + src.step * (i + n) + src.channels() * (j + m) + 0) * gaussianTemplate[m][n];
						accg += *(srcData + src.step * (i + n) + src.channels() * (j + m) + 1) * gaussianTemplate[m][n];
						accr += *(srcData + src.step * (i + n) + src.channels() * (j + m) + 2) * gaussianTemplate[m][n];
					}
				}
			}
			if (src.channels() == 1)
				*(dstData + dst.step * (i + 3) + dst.channels() * (j + 3)) = (int)acc;
			else
			{
				*(dstData + dst.step * (i + 3) + dst.channels() * (j + 3) + 0) = (int)accb;
				*(dstData + dst.step * (i + 3) + dst.channels() * (j + 3) + 1) = (int)accg;
				*(dstData + dst.step * (i + 3) + dst.channels() * (j + 3) + 2) = (int)accr;
			}
		}
	}
}

/*
  利用7*7高斯卷积核对uchar型一通道灰度图像进行平滑
  @param src 输入图像
  @param dst 输出图像
  @param sigma 高斯卷积核的sqrt(方差)
*/
void GaussianSmooth2D(const Mat& src, Mat& dst, double sigma)
{
	if (src.channels() != 1)
		return;

	//确保sigma为正数
	sigma = sigma > 0 ? sigma : 0;
	//高斯核矩阵的大小为(6*sigma+1)*(6*sigma+1)
	//ksize为奇数
	int ksize = cvRound(sigma * 3) * 2 + 1;

	if (ksize == 1)
	{
		src.copyTo(dst);
		return;
	}

	dst.create(src.size(), src.type());

	//计算高斯核矩阵
	double* kernel = new double[ksize * ksize];

	double scale = -0.5 / (sigma * sigma);
	const double PI = 3.141592653;
	double cons = -scale / PI;

	double sum = 0;

	for (int i = 0; i < ksize; i++)
	{
		for (int j = 0; j < ksize; j++)
		{
			int x = i - (ksize - 1) / 2;
			int y = j - (ksize - 1) / 2;
			kernel[i * ksize + j] = cons * exp(scale * (x * x + y * y));

			sum += kernel[i * ksize + j];
		}
	}
	//高斯核归一化
	for (int i = ksize * ksize - 1; i >= 0; i--)
	{
		*(kernel + i) /= sum;
	}

	uchar* srcData = src.data;
	uchar* dstData = dst.data;

	//图像卷积运算
	for (int j = 0; j < src.cols - ksize; j++)
	{
		for (int i = 0; i < src.rows - ksize; i++)
		{
			double acc = 0;

			for (int m = 0; m < ksize; m++)
			{
				for (int n = 0; n < ksize; n++)
				{
					acc += *(srcData + src.step * (i + n) + src.channels() * (j + m)) * kernel[m * ksize + n];
				}
			}

			*(dstData + dst.step * (i + (ksize - 1) / 2) + (j + (ksize - 1) / 2)) = (int)acc;
		}
	}
	delete[]kernel;
}

/*
  利用opencv官方函数对图像进行平滑
  @param src 输入图像
  @param dst 输出图像
  @param sigma 高斯卷积核的sqrt(方差)
*/
void GaussianSmooth(const Mat& src, Mat& dst, double sigma)
{
	GaussianBlur(src, dst, Size(0, 0), sigma, sigma);
}

/*
  创建第0组第0层图像，这里假设其尺度为sigma
  @param src 输入图像
  @param dst 输出图像
  @param sigma -1层（第0组第0层）的尺度
*/
void CreateInitSmoothGray(const Mat& src, Mat& dst, double sigma = SIGMA)
{
	Mat gray, up;

	ConvertToGray(src, gray);
	UpSample(gray, up);

	//sigma是-1层的尺度，sigma_init是高斯卷积核的sprt(方差)
	double  sigma_init = sqrt(sigma * sigma - (INIT_SIGMA * 2) * (INIT_SIGMA * 2));

	GaussianSmooth(up, dst, sigma_init);
}

/*
  创建高斯尺度空间
  @param src 输入图像
  @param gauss_pyr 高斯金字塔
  @param octaves 高斯金字塔的组数
  @param intervals 有效极值点检测的层数
  @param sigma -1层（第0组第0层）的尺度
*/
void GaussianPyramid(const Mat& src, vector<Mat>& gauss_pyr, int octaves, int intervals = INTERVALS, double sigma = SIGMA)
{
	//构造存储组内层间高斯平滑所需要的因子的数组，之后每一组都是用这一数组去生成，仔细想一想原因！！！
	//sqrt(sig_total * sig_total - sig_prev * sig_prev)
	double* sigmas = new double[intervals + 3];
	double k = pow(2.0, 1.0 / intervals);

	sigmas[0] = sigma;

	double sig_prev, sig_total;
	for (int i = 1; i < intervals + 3; i++)
	{
		sig_prev = pow(k, i - 1) * sigma;
		sig_total = sig_prev * k;
		sigmas[i] = sqrt(sig_total * sig_total - sig_prev * sig_prev);
	}

	for (int o = 0; o < octaves; o++)
	{
		//每组多三层
		for (int i = 0; i < intervals + 3; i++)
		{
			Mat mat;
			if (o == 0 && i == 0)
			{
				src.copyTo(mat);
			}
			else if (i == 0)
			{
				//			DownSample(gauss_pyr[o*(intervals+3)-2], mat);  //error
								//前一组高斯图像的倒数第三层（正数第四层）
								//如图像下标为：
								//0 1 2 3  4  5  //o=0
								//6 7 8 9 10 11  //o=1
								//...
								//第一组第一张图(下标为6)的图像是0组下标为3的图像降采样得来
				DownSample(gauss_pyr[(o - 1) * (intervals + 3) + intervals], mat);
			}
			else
			{
				//每组中下一层由上一层高斯模糊得到（所用高斯卷积核的大小为sqrt(sig_late * sig_late - sig_prev * sig_prev)）
				GaussianSmooth(gauss_pyr[o * (intervals + 3) + i - 1], mat, sigmas[i]);
			}
			gauss_pyr.push_back(mat);
		}
	}
	delete[] sigmas;
}

/*
  由高斯金字塔同一组的相邻两张图像生成差分图像
  @param a 输入图像1
  @param b 输入图像2
  @param c 输出图像（差分图像）
*/
void Sub(const Mat& a, const Mat& b, Mat& c)
{
	if (a.rows != b.rows || a.cols != b.cols || a.type() != b.type())
		return;
	if (!c.empty())
		return;
	c.create(a.size(), a.type());

	pixel_t* ap = (pixel_t*)a.data;
	pixel_t* ap_end = (pixel_t*)a.dataend;
	pixel_t* bp = (pixel_t*)b.data;
	pixel_t* cp = (pixel_t*)c.data;

	int step = a.step / sizeof(pixel_t);

	//while的使用使代码更加简洁，值得学习
	while (ap != ap_end)
	{
		*cp++ = *ap++ - *bp++;
	}
}

/*
  生成高斯差分金字塔
  @param gauss_pyr 高斯金字塔
  @param dog_pyr 高斯差分金字塔
  @param octaves 高斯差分金字塔的组数
  @param intervals 有效极值点检测的层数
*/
void DogPyramid(const vector<Mat>& gauss_pyr, vector<Mat>& dog_pyr, int octaves, int intervals = INTERVALS)
{
	for (int o = 0; o < octaves; o++)
	{
		for (int i = 1; i < intervals + 3; i++)
		{
			Mat mat;
			//生成的差分图像的尺度与gauss_pyr[o*(intervals + 3) + i - 1]的尺度是一样的，这点很重要！！！
			Sub(gauss_pyr[o * (intervals + 3) + i], gauss_pyr[o * (intervals + 3) + i - 1], mat);
			dog_pyr.push_back(mat);
		}
	}
}

/*
  判断某像素在其周围3*3*3邻域内是否为极值
  @param x 像素的x坐标（图像坐标系下）
  @param y 像素的y坐标（图像坐标系下）
  @param dog_pyr 高斯差分金字塔
  @param index 该像素所在图层的索引
  @return 如果是极值则返回true，不是则返回false
*/
bool isExtremum(int x, int y, const vector<Mat>& dog_pyr, int index)
{
	pixel_t* data = (pixel_t*)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	if (val > 0)
	{
		//遍历层
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(data[0]);
			//遍历行
			for (int j = -1; j <= 1; j++)
			{
				//遍历列
				for (int k = -1; k <= 1; k++)
				{
					//检查最大极值
					if (val < *((pixel_t*)dog_pyr[index + i].data + stp * (y + j) + (x + k)))
					{
						return false;
					}
				}
			}
		}
	}
	else
	{
		//遍历层
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(data[0]);
			//int stp = dog_pyr[index + i].step / sizeof(pixel_t);
			//其与上面一行代码意义一样，都是获取元素的字节数，一个是直接访问元素，一个是访问double类型

			//遍历行
			for (int j = -1; j <= 1; j++)
			{
				//遍历列
				for (int k = -1; k <= 1; k++)
				{
					//检查最小极值
					if (val > *((pixel_t*)dog_pyr[index + i].data + stp * (y + j) + (x + k)))
					{
						return false;
					}
				}
			}
		}
	}

	return true;
}

bool isExtremum_nointerval(int x, int y, const vector<Mat>& dog_pyr, int index)
{
	pixel_t* data = (pixel_t*)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	if (val > 0)
	{
		int stp = dog_pyr[index].step / sizeof(data[0]);
		//遍历行
		for (int j = -1; j <= 1; j++)
		{
			//遍历列
			for (int k = -1; k <= 1; k++)
			{
				//检查最大极值
				if (val < *((pixel_t*)dog_pyr[index].data + stp * (y + j) + (x + k)))
				{
					return false;
				}
			}
		}
	}

	else
	{
		int stp = dog_pyr[index].step / sizeof(data[0]);
		//int stp = dog_pyr[index + i].step / sizeof(pixel_t);
		//其与上面一行代码意义一样，都是获取元素的字节数，一个是直接访问元素，一个是访问double类型

		//遍历行
		for (int j = -1; j <= 1; j++)
		{
			//遍历列
			for (int k = -1; k <= 1; k++)
			{
				//检查最小极值
				if (val > *((pixel_t*)dog_pyr[index].data + stp * (y + j) + (x + k)))
				{
					return false;
				}
			}
		}
	}

	return true;
}

//访问图像的x、y位置的像素（x表示横坐标，y表示纵坐标）
#define DAt(x, y) (*(data+(y)*step+(x)))

/*
  消除边缘响应
  Tr_h * Tr_h / Det_h的值在两个特征值相等时最小；值越大，说明两个特征值的比值越大。
  即在某一个方向的梯度值越大，而在另一个方向的梯度值越小，而边缘恰恰就是这种情况。
  所以为了剔除边缘响应点，需要让该比值小于一定的阈值。
  @param x 像素的横坐标
  @param y 像素的纵坐标
  @param dog_pyr 高斯差分金字塔
  @param index 该像素所在图层的索引
  @param r 消除边缘响应所需的阈值，Lowe建议为10
  @return 如果小于阈值则返回true，否则返回false
*/
bool passEdgeResponse(int x, int y, const vector<Mat>& dog_pyr, int index, double r = RATIO)
{
	pixel_t* data = (pixel_t*)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	double Dxx, Dyy, Dxy;
	double Tr_h, Det_h;

	//hessian矩阵
	//	   _ 	    _
	//    | Dxx  Dxy |
	// H =|			 |
	//	  |_Dxy  Dyy_|
	//
	Dxx = DAt(x + 1, y) + DAt(x - 1, y) - 2 * val;
	Dyy = DAt(x, y + 1) + DAt(x, y - 1) - 2 * val;
	Dxy = (DAt(x + 1, y + 1) + DAt(x - 1, y - 1) - DAt(x - 1, y + 1) - DAt(x + 1, y - 1)) / 4.0;

	Tr_h = Dxx + Dyy;
	Det_h = Dxx * Dyy - Dxy * Dxy;

	if (Det_h <= 0)
		return false;

	//小于阈值才认为其是关键点
	if (Tr_h * Tr_h / Det_h < (r + 1) * (r + 1) / r)
		return true;

	return false;
}

//访问金字塔索引为index层的x、y位置的像素
double PyrAt(const vector<Mat>& pyr, int index, int x, int y)
{
	pixel_t* data = (pixel_t*)pyr[index].data;
	int step = pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	return val;
}

#define At(index, x, y) (PyrAt(dog_pyr, (index), (x), (y)))

/*
  生成一阶导数数组（1*3）
  @param x 像素的横坐标（图像坐标系下）
  @param y 像素的纵坐标（图像坐标系下）
  @param dog_pyr 高斯差分金字塔
  @param index 该像素所在图层的索引
  @param dx 存储一阶导数的数组
*/
void DerivativeOf3D(int x, int y, const vector<Mat>& dog_pyr, int index, double* dx)
{
	double Dx = (At(index, x + 1, y) - At(index, x - 1, y)) / 2.0;
	double Dy = (At(index, x, y + 1) - At(index, x, y - 1)) / 2.0;
	double Ds = (At(index + 1, x, y) - At(index - 1, x, y)) / 2.0;

	dx[0] = Dx;
	dx[1] = Dy;
	dx[2] = Ds;
}

//访问Hessian矩阵的i、j位置的元素
//Hessian矩阵数据用一维数组存储
#define Hat(i, j) (*(H+(i)*3 + (j)))

/*
  生成Hessian矩阵（3*3）
  @param x 像素的横坐标（图像坐标系下）
  @param y 像素的纵坐标（图像坐标系下）
  @param dog_pyr 高斯差分金字塔
  @param index 该像素所在图层的索引
  @param H 存储Hessian矩阵的数组
*/
void Hessian3D(int x, int y, const vector<Mat>& dog_pyr, int index, double* H)
{
	double val, Dxx, Dyy, Dss, Dxy, Dxs, Dys;

	val = At(index, x, y);

	Dxx = At(index, x + 1, y) + At(index, x - 1, y) - 2 * val;
	Dyy = At(index, x, y + 1) + At(index, x, y - 1) - 2 * val;
	Dss = At(index + 1, x, y) + At(index - 1, x, y) - 2 * val;

	Dxy = (At(index, x + 1, y + 1) + At(index, x - 1, y - 1)
		- At(index, x + 1, y - 1) - At(index, x - 1, y + 1)) / 4.0;

	Dxs = (At(index + 1, x + 1, y) + At(index - 1, x - 1, y)
		- At(index - 1, x + 1, y) - At(index + 1, x - 1, y)) / 4.0;

	Dys = (At(index + 1, x, y + 1) + At(index - 1, x, y - 1)
		- At(index + 1, x, y - 1) - At(index - 1, x, y + 1)) / 4.0;

	Hat(0, 0) = Dxx;
	Hat(1, 1) = Dyy;
	Hat(2, 2) = Dss;

	Hat(1, 0) = Hat(0, 1) = Dxy;
	Hat(2, 0) = Hat(0, 2) = Dxs;
	Hat(2, 1) = Hat(1, 2) = Dys;
}

//访问Hessian矩阵的逆的i、j位置的元素
//Hessian矩阵的逆的数据用一维数组存储
#define HIat(i, j) (*(H_inve+(i)*3 + (j)))

/*
  生成Hessian矩阵的逆（3*3）
  @param x 像素的横坐标（图像坐标系下）
  @param y 像素的纵坐标（图像坐标系下）
  @param dog_pyr 高斯差分金字塔
  @param index 该像素所在图层的索引
  @param H_inve 存储Hessian矩阵的逆的数组
  @return 如果可以求逆则返回true，否则返回false
*/
bool Inverse3D(const double* H, double* H_inve)
{
	//A=|H|
	//		 / A00 A01 A02 \
	//若H =  | A10 A11 A12 |
	//		 \ A20 A21 A22 /
	//则 行列式|H|=A00*A11*A22+A01*A12*A20+A02*A10*A21
	//	    -A00*A12*A21-A01*A10*A22-A02*A11*A20
	//

	double A = Hat(0, 0) * Hat(1, 1) * Hat(2, 2)
		+ Hat(0, 1) * Hat(1, 2) * Hat(2, 0)
		+ Hat(0, 2) * Hat(1, 0) * Hat(2, 1)
		- Hat(0, 0) * Hat(1, 2) * Hat(2, 1)
		- Hat(0, 1) * Hat(1, 0) * Hat(2, 2)
		- Hat(0, 2) * Hat(1, 1) * Hat(2, 0);

	//没有逆矩阵
	if (fabs(A) < 1e-10)
		return false;

	//三阶逆矩阵运算公式（伴随矩阵法）：
	//		 / a b c \				    / ei-hf -(bi-ch) bf-ce\
	//若A =  | d e f |   则A(-1) =1/|H|*| fg-id -(cg-ia) cd-af |
	//		 \ g h i /				    \ dh-ge -(ah-gb) ae-bd/

	HIat(0, 0) = Hat(1, 1) * Hat(2, 2) - Hat(2, 1) * Hat(1, 2);
	HIat(0, 1) = -(Hat(0, 1) * Hat(2, 2) - Hat(2, 1) * Hat(0, 2));
	HIat(0, 2) = Hat(0, 1) * Hat(1, 2) - Hat(0, 2) * Hat(1, 1);

	HIat(1, 0) = Hat(1, 2) * Hat(2, 0) - Hat(2, 2) * Hat(1, 0);
	HIat(1, 1) = -(Hat(0, 2) * Hat(2, 0) - Hat(0, 0) * Hat(2, 2));
	HIat(1, 2) = Hat(0, 2) * Hat(1, 0) - Hat(0, 0) * Hat(1, 2);

	HIat(2, 0) = Hat(1, 0) * Hat(2, 1) - Hat(1, 1) * Hat(2, 0);
	HIat(2, 1) = -(Hat(0, 0) * Hat(2, 1) - Hat(0, 1) * Hat(2, 0));
	HIat(2, 2) = Hat(0, 0) * Hat(1, 1) - Hat(0, 1) * Hat(1, 0);

	for (int i = 0; i < 9; i++)
	{
		//i = 0时，指针指向数组的初始位置
		*(H_inve + i) /= A;
	}
	return true;
}

/*
  生成亚像素级插值后坐标的偏移量的数组（1*3）
  @param x 像素的横坐标（图像坐标系下）
  @param y 像素的纵坐标（图像坐标系下）
  @param dog_pyr 高斯差分金字塔
  @param index 该像素所在图层的索引
  @param offset_x 存储亚像素级插值后坐标的偏移量的数组
*/
void GetOffsetX(int x, int y, const vector<Mat>& dog_pyr, int index, double* offset_x)
{
	//x^ = -H^(-1) * dx; dx = (Dx, Dy, Ds)^T
	double H[9], H_inve[9] = { 0 };
	Hessian3D(x, y, dog_pyr, index, H);
	Inverse3D(H, H_inve);
	double dx[3];
	DerivativeOf3D(x, y, dog_pyr, index, dx);

	for (int i = 0; i < 3; i++)
	{
		offset_x[i] = 0.0;
		//先矩阵运算
		for (int j = 0; j < 3; j++)
		{
			offset_x[i] += H_inve[i * 3 + j] * dx[j];
		}
		//再求相反数
		offset_x[i] = -offset_x[i];
	}
}

/*
  生成Dx，用来进行对比度判断
  @param x 像素的横坐标（图像坐标系下）
  @param y 像素的纵坐标（图像坐标系下）
  @param dog_pyr 高斯差分金字塔
  @param index 该像素所在图层的索引
  @param offset_x 存储亚像素级插值后坐标的偏移量的数组
  @return Dx
*/
double GetFabsDx(int x, int y, const vector<Mat>& dog_pyr, int index, const double* offset_x)
{
	//|D(x^)|=D + 0.5 * dx * offset_x; dx=(Dx, Dy, Ds)^T
	double dx[3];
	DerivativeOf3D(x, y, dog_pyr, index, dx);

	double term = 0.0;
	for (int i = 0; i < 3; i++)
		term += dx[i] * offset_x[i];

	pixel_t* data = (pixel_t*)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	return fabs(val + 0.5 * term);
}

/*
  进行亚像素级插值
  @param x 像素的横坐标（图像坐标系下）
  @param y 像素的纵坐标（图像坐标系下）
  @param dog_pyr 高斯差分金字塔
  @param index 该像素所在图层的索引
  @param ocatave 极值点所在的组
  @param interval 极值点所在组的层
  @param dxthreshold 判断Dx用的阈值，|D(x)| < 0.03 Lowe 2004
  @return Keypoint 关键点坐标等信息
*/
Keypoint* InterploationExtremum(int x, int y, const vector<Mat>& dog_pyr, int index, int octave, int interval, double dxthreshold = DXTHRESHOLD)
{
	double offset_x[3] = { 0 };
	//将dog_pyr[index]定义为const，防止被窜改
	const Mat& mat = dog_pyr[index];

	//当前图层所在的索引，可由octave和interval计算出来
	int idx = index;

	//当前图层处于当前组的哪一层
	int intvl = interval;
	//插值次数变量
	int i = 0;
	while (i < MAX_INTERPOLATION_STEPS)
	{
		GetOffsetX(x, y, dog_pyr, idx, offset_x);
		//Accurate keypoint localization.  Lowe
		//如果offset_x 的任一维度大于0.5，it means that the extremum lies closer to a different sample point.
		if (fabs(offset_x[0]) < 0.5 && fabs(offset_x[1]) < 0.5 && fabs(offset_x[2]) < 0.5)
			//如果泰勒插值五次后三个坐标的偏移量都小于0.5，说明已经找到特征点，则退出迭代
			break;

		//由上面得到的偏移量重新定义插值中心的坐标位置
		x += cvRound(offset_x[0]);
		y += cvRound(offset_x[1]);
		interval += cvRound(offset_x[2]);

		//修正索引idx，我们使用idx来访问DoG金字塔
		idx = index - intvl + interval;

		//此处保证检测边时 x+1,y+1和x-1, y-1有效
		if (interval < 1 || interval > INTERVALS ||
			x >= mat.cols - 1 || x < 2 ||
			y >= mat.rows - 1 || y < 2)
		{
			return NULL;
		}

		i++;
	}

	//窜改失败，确认是否大于迭代次数
	if (i >= MAX_INTERPOLATION_STEPS)
		return NULL;

	//rejecting unstable extrema，剔除不稳定的极值点
	//|D(x^)| < 0.03取经验值，此时特征点会非常多
	if (GetFabsDx(x, y, dog_pyr, idx, offset_x) < dxthreshold / INTERVALS)
	{
		return NULL;
	}

	//将当前点存入特征点中
	Keypoint* keypoint = new Keypoint;

	keypoint->x = x;
	keypoint->y = y;

	keypoint->offset_x = offset_x[0];
	keypoint->offset_y = offset_x[1];

	keypoint->interval = interval;
	keypoint->offset_interval = offset_x[2];

	keypoint->octave = octave;

	keypoint->dx = (x + offset_x[0]) * pow(2.0, octave);
	keypoint->dy = (y + offset_x[1]) * pow(2.0, octave);

	return keypoint;
}

/*
  进行极值点的检测，剔除低对比度的点，判断是否是极值点，并进行亚像素级插值，剔除边缘响应点
  @param dog_pyr 高斯差分金字塔
  @param extrma 存储关键点的vector
  @param ocataves 高斯差分金字塔组数
  @param intervals 有效极值点检测的层数
*/
void DetectionLocalExtrema(const vector<Mat>& dog_pyr, vector<Keypoint>& extrema, int octaves, int intervals = INTERVALS)
{
	long int dd = 0, cc1 = 0, cc2 = 0, cc3 = 0, cc0 = 0, cc00 = 0;

	double thresh = 0.5 * DXTHRESHOLD / intervals;
	for (int o = 0; o < octaves; o++)
	{
		//第一层和最后一层极值忽略，这里再次思考这几层的尺度！！！
		for (int i = 1; i < (intervals + 2) - 1; i++)
		{
			//将当前图层的索引用octave和interval表示出来
			int index = o * (intervals + 2) + i;

			pixel_t* data = (pixel_t*)dog_pyr[index].data;
			int step = dog_pyr[index].step / sizeof(data[0]);

			for (int y = IMG_BORDER; y < dog_pyr[index].rows - IMG_BORDER; y++)
			{
				for (int x = IMG_BORDER; x < dog_pyr[index].cols - IMG_BORDER; x++)
				{
					//统计判断了多少个像素点?
					cc00++;

					pixel_t val = *(data + y * step + x);
					//剔除小于阈值的点
					if (fabs(val) > thresh)
					{
						//统计判断了多少个像素点大于给定的阈值?
						cc0++;
						if (isExtremum(x, y, dog_pyr, index))
						{
							//统计判断了多少个像素点是局部极大值?
							cc1++;
							Keypoint* extrmum = InterploationExtremum(x, y, dog_pyr, index, o, i);

							if (extrmum)
							{
								//统计判断了多少个像素点进行了亚像素级插值?
								cc2++;

								if (passEdgeResponse(extrmum->x, extrmum->y, dog_pyr, index))
								{
									extrmum->val = *(data + extrmum->y * step + extrmum->x);

									//统计判断了多少个像素点进行了消除边缘点响应?
									cc3++;
									extrema.push_back(*extrmum);
								}

								delete extrmum;
							}
						}
					}
				}
			}
		}
	}
	std::cout << "cc00: " << cc00 << ", cc0: " << cc0 << ", cc1: " << cc1 << ", cc2: " <<
		cc2 << ", cc3: " << cc3 << std::endl;
	std::cout << "0.5 * DXTHRESHOLD / intervals: " << thresh << std::endl;
}

void DetectionLocalExtrema_nointerval(const vector<Mat>& dog_pyr, vector<Keypoint>& extrema, int octaves, int intervals = INTERVALS)
{
	long int dd = 0, cc1 = 0, cc2 = 0, cc3 = 0, cc0 = 0, cc00 = 0;

	double thresh = 0.5 * DXTHRESHOLD / intervals;
	for (int o = 0; o < octaves; o++)
	{
		//第一层和最后一层极值忽略，这里再次思考这几层的尺度！！！
		for (int i = 1; i < (intervals + 2) - 1; i++)
		{
			//将当前图层的索引用octave和interval表示出来
			int index = o * (intervals + 2) + i;

			pixel_t* data = (pixel_t*)dog_pyr[index].data;
			int step = dog_pyr[index].step / sizeof(data[0]);

			for (int y = IMG_BORDER; y < dog_pyr[index].rows - IMG_BORDER; y++)
			{
				for (int x = IMG_BORDER; x < dog_pyr[index].cols - IMG_BORDER; x++)
				{
					//统计判断了多少个像素点?
					cc00++;

					pixel_t val = *(data + y * step + x);
					//剔除小于阈值的点
					if (fabs(val) > thresh)
					{
						//统计判断了多少个像素点大于给定的阈值?
						cc0++;
						if (isExtremum_nointerval(x, y, dog_pyr, index))
						{
							//统计判断了多少个像素点是局部极大值?
							cc1++;
							Keypoint* extrmum = InterploationExtremum(x, y, dog_pyr, index, o, i);

							if (extrmum)
							{
								//统计判断了多少个像素点进行了亚像素级插值?
								cc2++;

								if (passEdgeResponse(extrmum->x, extrmum->y, dog_pyr, index))
								{
									extrmum->val = *(data + extrmum->y * step + extrmum->x);

									//统计判断了多少个像素点进行了消除边缘点响应?
									cc3++;
									extrema.push_back(*extrmum);
								}

								delete extrmum;
							}
						}
					}
				}
			}
		}
	}
	std::cout << "cc00: " << cc00 << ", cc0: " << cc0 << ", cc1: " << cc1 << ", cc2: " <<
		cc2 << ", cc3: " << cc3 << std::endl;
	std::cout << "0.5 * DXTHRESHOLD / intervals: " << thresh << std::endl;
}

/*
  计算极值点所在层的尺度
  @param features 存储关键点的vector
  @param sigam -1层（第0组第0层）的尺度
  @param intervals 有效极值点检测的层数
*/
void CalculateScale(vector<Keypoint>& features, double sigma = SIGMA, int intervals = INTERVALS)
{
	double intvl = 0;
	for (int i = 0; i < features.size(); i++)
	{
		intvl = features[i].interval + features[i].offset_interval;
		//相对于第0组第0层的尺度，用于恢复关键点坐标和尺度
		features[i].scale = sigma * pow(2.0, features[i].octave + intvl / intervals);
		//相对于所在组第0层的尺度，用于确定统计关键点梯度方向直方图的邻域的半径
		features[i].octave_scale = sigma * pow(2.0, intvl / intervals);
	}
}

/*
  对扩大的图像进行特征缩放，缩放到输入图像的尺度
  @param features 存储特征点的vector
*/
void HalfFeatures(vector<Keypoint>& features)
{
	for (int i = 0; i < features.size(); i++)
	{
		//dx、dy是坐标是相对于第0组第0层的坐标系，需要恢复到输入图像的尺度
		features[i].dx /= 2;
		features[i].dy /= 2;

		//scale是坐标是相对于第0组第0层的尺度，需要恢复到输入图像的尺度
		features[i].scale /= 2;
	}
}

/*
  计算gauss图像中x、y像素点的幅值和梯度方向
  @param gauss 高斯金字塔中的图像
  @param x 像素的x坐标（图像坐标系下）
  @param y 像素的y坐标（图像坐标系下）
  @param mag 幅值
  @param ori 梯度方向
  @return 没有触碰到边缘则返回true，否则返回false
*/
bool CalcGradMagOri(const Mat& gauss, int x, int y, double& mag, double& ori)
{
	if (x > 0 && x < gauss.cols - 1 && y > 0 && y < gauss.rows - 1)
	{
		pixel_t* data = (pixel_t*)gauss.data;
		int step = gauss.step / sizeof(*data);

		double dx = *(data + step * y + (x + 1)) - (*(data + step * y + (x - 1)));
		double dy = *(data + step * (y + 1) + x) - (*(data + step * (y - 1) + x));

		mag = sqrt(dx * dx + dy * dy);

		//atan2返回[-Pi, -Pi]的弧度值
		//
		//				0.5*PI
		//				|
		//		PI		|
		//     ----------------- 0
		//		-PI		|
		//				|
		//				-0.5*PI
		//
		ori = atan2(dy, dx);
		return true;
	}
	else
		return false;
}

/*
  计算极值点的梯度方向直方图
  @param gauss 高斯金字塔中的图像
  @param x 像素的x坐标（图像坐标系下）
  @param y 像素的y坐标（图像坐标系下）
  @param bins 梯度方向直方图的bins
  @param radius 统计梯度方向直方图所需要的邻域半径
  @return 返回梯度方向直方图
*/
double* CalculateOrientationHistogram(const Mat& gauss, int x, int y, int bins, int radius, double sigma)
{
	//一般为36bin
	double* hist = new double[bins];

	//梯度方向直方图赋初值
	for (int i = 0; i < bins; i++)
		*(hist + i) = 0.0;

	double mag, ori;
	double weight;

	int bin;
	const double PI2 = 2.0 * CV_PI;

	double econs = -1.0 / (2.0 * sigma * sigma);

	//radius: cvRound(ORI_WINDOW_RADIUS*extrema[i].octave_scale) = cvRound(3*1.5*extrema[i].octave_scale)
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			if (CalcGradMagOri(gauss, x + i, y + j, mag, ori))
			{
				//sigam: ORI_SIGMA_TIMES*extrema[i].octave_scale = 1.5*extrema[i].octave_scale
				weight = exp((i * i + j * j) * econs);

				//使用Pi-ori将ori转换到[0,2*PI]之间
				//
				//
				//	转换前：									转换后：
				//
				//				0.5*PI									0.5*PI
				//				|										|
				//		PI		|								0		|
				//     ----------------- 0						----------------- PI
				//		-PI		|								2*PI	|
				//				|										|
				//				-0.5*PI									1.5*PI
				//
				//
				bin = cvRound(bins * (CV_PI - ori) / PI2);
				bin = bin < bins ? bin : 0;

				hist[bin] += mag * weight;
			}
		}
	}

	return hist;
}

/*
  对梯度方向直方图进行平滑
  高斯平滑，模板为{0.25, 0.5, 0.25}
  可以理解为将原来的hist做平滑后的值赋给新的hist，这里实现方法很巧妙！！！
  @param hist 梯度方向直方图
  @param n 梯度方向直方图的bins
*/
void GaussSmoothOriHist(double* hist, int n)
{
	double prev = hist[n - 1], temp, h0 = hist[0];

	for (int i = 0; i < n; i++)
	{
		temp = hist[i];
		hist[i] = 0.25 * prev + 0.5 * hist[i] +
			0.25 * (i + 1 >= n ? h0 : hist[i + 1]);
		prev = temp;
	}
}

/*
  求梯度方向直方图的主方向
  @param hist 梯度方向直方图
  @param n 梯度方向直方图的bins
  @return 返回所有bin中的最大值
*/
double DominantDirection(double* hist, int n)
{
	double maxd = hist[0];
	for (int i = 1; i < n; i++)
	{
		if (hist[i] > maxd)
			maxd = hist[i];
	}
	return maxd;
}

/*
  复制特征点
  一个特征点出现辅方向时，也要记作一个新的特征点，因此直接copy一份，再修改方向，操作简便
  @param src 输入特征点
  @param dst 输出特征点
*/
void CopyKeypoint(const Keypoint& src, Keypoint& dst)
{
	dst.dx = src.dx;
	dst.dy = src.dy;

	dst.interval = src.interval;
	dst.octave = src.octave;
	dst.octave_scale = src.octave_scale;
	dst.offset_interval = src.offset_interval;

	dst.offset_x = src.offset_x;
	dst.offset_y = src.offset_y;

	dst.ori = src.ori;
	dst.scale = src.scale;
	dst.val = src.val;
	dst.x = src.x;
	dst.y = src.y;
}

//抛物线插值
#define Parabola_Interpolate(l, c, r) (0.5*((l)-(r))/((l)-2.0*(c)+(r)))

/*
  对平滑后的梯度方向直方图进行抛物线插值，获取更加准确的方向
  @param keypoint 输入特征点
  @param features 输出特征点的vector
  @param hist 梯度方向直方图
  @param mag_thr 梯度方向直方图中如果有某个bin的值大于主bin值的80%，则可以形成辅方向
*/
void CalcOriFeatures(const Keypoint& keypoint, vector<Keypoint>& features, const double* hist, int n, double mag_thr)
{
	double bin, PI2 = CV_PI * 2.0;

	int l, r;
	for (int i = 0; i < n; i++)
	{
		l = (i == 0) ? n - 1 : i - 1;
		r = (i + 1) % n;

		//hist[i]是极值，对于大于主方向80%能量值的bin需要认定为辅方向
		//mag_thr: highest_peak*ORI_PEAK_RATIO = 0.8*highest_peak
		if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
		{
			bin = i + Parabola_Interpolate(hist[l], hist[i], hist[r]);
			//从左到右理解
			bin = (bin < 0) ? (bin + n) : (bin >= n ? (bin - n) : bin);

			//new_key与keypoint只是方向不一样
			Keypoint new_key;
			CopyKeypoint(keypoint, new_key);

			//主方向、辅方向都考虑进去，形成新的特征点，因此特征点的数量要多于关键点的数量
			new_key.ori = ((PI2 * bin) / n) - CV_PI;
			features.push_back(new_key);
		}
	}
}

/*
  对特征点进行方向分配，计算特征点的梯度方向直方图，进行梯度方向直方图的平滑，确定主方向，确定主方向的精确角度和辅方向的精确角度
  @param extreme 输入特征点vector
  @param features 输出特征点的vector
  @param gauss_pyr 高斯金字塔
*/
void OrientationAssignment(vector<Keypoint>& extrema, vector<Keypoint>& features, const vector<Mat>& gauss_pyr)
{
	int n = extrema.size();
	double* hist;

	for (int i = 0; i < n; i++)
	{
		hist = CalculateOrientationHistogram(gauss_pyr[extrema[i].octave * (INTERVALS + 3) + extrema[i].interval],
			extrema[i].x, extrema[i].y, ORI_HIST_BINS, cvRound(ORI_WINDOW_RADIUS * extrema[i].octave_scale),
			ORI_SIGMA_TIMES * extrema[i].octave_scale);

		for (int j = 0; j < ORI_SMOOTH_TIMES; j++)
			GaussSmoothOriHist(hist, ORI_HIST_BINS);

		double highest_peak = DominantDirection(hist, ORI_HIST_BINS);

		CalcOriFeatures(extrema[i], features, hist, ORI_HIST_BINS, highest_peak * ORI_PEAK_RATIO);

		delete[] hist;
	}
}

/*
  三线性插值
  @param hist 特征描绘子数组
  @param xbin 像素的x坐标（图像坐标系下）
  @param ybin 像素的y坐标（图像坐标系下）
  @param obin 像素的index
  @param mag 梯度方向
  @param bins DESCR_HIST_BINS = 8
  @param d DESCR_WINDOW_WIDTH = 4
*/
void InterpHistEntry(double*** hist, double xbin, double ybin, double obin, double mag, int bins, int d)
{
	//代表正方体中(0, 0, 0)
	int r0, c0, o0;

	//代表正方体中(0, 0, 0)点对角位置的点(1, 1, 1)
	int rb, cb, ob;

	//代表像素点在正方体坐标系中的位置，取值范围[0, 1]
	double d_r, d_c, d_o;

	//临时变量用来表示像素点对某个顶点的权重，注意这里是一步一步计算的
	//权重与距离成反比：weight(r) = 1 - r
	double v_r, v_c, v_o;

	//临时变量用来访问正方体的八个顶点
	int r, c, o;

	double** row, * h;

	//向下取整，r0、c0、o0为三维坐标的整数部分，表示属于哪个正方体
	r0 = cvFloor(ybin);
	c0 = cvFloor(xbin);
	o0 = cvFloor(obin);

	//d_r、d_c、d_o为三维坐标的小数部分，即正方体内C点的坐标
	d_r = ybin - r0;
	d_c = xbin - c0;
	d_o = obin - o0;

	/*
		做插值：
		xbin,ybin,obin:种子点所在子窗口的位置和方向
		所有种子点都将落在4*4的窗口中
		r0,c0取不大于xbin，ybin的正整数
		r0,c0只能取到0,1,2
		xbin,ybin在(-1, 2)
		r0取不大于xbin的正整数时。
		r0+0 <= xbin <= r0+1
		mag在区间[r0,r1]上做插值
		obin同理
	*/

	for (r = 0; r <= 1; r++)
	{
		//借此来访问正方体的八个顶点，例如点（-0.5，-0.5，6.3）
		//则其不必计算对（-1，-1，？）各点的权重，只需计算对（0，0，？）点的权重
		//又或者如点（ 3.5， 3.5，6.3）
		//则其不必计算对（4，4，？）各点的权重，只需计算对（3，3，？）点的权重
		rb = r0 + r;
		//判断row方向上是否超出统计区域，rb = {0, 1, 2, 3}
		if (rb >= 0 && rb < d)
		{
			//相对row的权重，r == 0则计算其左上角点的权重，r != 0则计算其右下角的点的权重，权重与距离成反比
			v_r = mag * ((r == 0) ? 1.0 - d_r : d_r);
			row = hist[rb];
			for (c = 0; c <= 1; c++)
			{
				cb = c0 + c;
				//判断col方向上是否超出统计区域，cb = {0, 1, 2, 3}
				if (cb >= 0 && cb < d)
				{
					//相对row的权重*相对col的权重
					v_c = v_r * ((c == 0) ? 1.0 - d_c : d_c);
					h = row[cb];
					for (o = 0; o <= 1; o++)
					{
						//因为梯度方向是循环的，所以直接用取余的方法
						//例如点（-0.5，-0.5，7.2），其对 7 和 0 ori层都会产生权重
						//所以不用对其进行是否超限的判断
						ob = (o0 + o) % bins;
						//相对row的权重*相对col的权重*相对ori的权重
						v_o = v_c * ((o == 0) ? 1.0 - d_o : d_o);
						h[ob] += v_o;
					}
				}
			}
		}
	}
}

/*
  计算特征描绘子数组
  @param gauss gauss图像
  @param x 像素的x坐标（图像坐标系下）
  @param y 像素的y坐标（图像坐标系下）
  @param octave_scale gauss图像相对于组内第0层的尺度
  @param ori 计算特征描绘子邻域需要旋转的角度
  @param bins 特征描绘子梯度方向直方图的bins
  @param width 邻域划分为4*4的个方格
  在实际应用中，我们是先以特征点为圆心，以(DESCR_SCALE_ADJUST * octave_scale*sqrt(2.0) * (width + 1)) / 2.0为半径，
  计算该圆内所有像素的梯度幅角和高斯加权后的梯度幅值，然后再得到这些幅值和幅角所对应的像素在旋转以后新的坐标位置。
*/
double*** CalculateDescrHist(const Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width)
{
	double*** hist = new double** [width];

	//申请空间并初始化，4*4*8数组
	for (int i = 0; i < width; i++)
	{
		hist[i] = new double* [width];
		for (int j = 0; j < width; j++)
		{
			hist[i][j] = new double[bins];
		}
	}

	for (int r = 0; r < width; r++)
		for (int c = 0; c < width; c++)
			for (int o = 0; o < bins; o++)
				hist[r][c][o] = 0.0;

	double cos_ori = cos(ori);
	double sin_ori = sin(ori);

	//高斯权值，sigma等于描述子窗口宽度（4）的一半
	double sigma = 0.5 * width;
	double conste = -1.0 / (2 * sigma * sigma);

	double PI2 = CV_PI * 2;

	//每个子区域(种子所在的小方格)的宽
	double sub_hist_width = DESCR_SCALE_ADJUST * octave_scale;

	//邻域半径，+0.5取四舍五入，实际上不需要，因为不加0.5产生的区域用于运算已经足够（已经考虑了旋转带来的影响）
	//int radius = (sub_hist_width*sqrt(2.0)*(width + 1)) / 2.0 + 0.5;
	int radius = (sub_hist_width * sqrt(2.0) * (width + 1)) / 2.0;

	double grad_ori, grad_mag;
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			//顺时针旋转，现在需要计算的点已经落在了4*4窗口中，但是坐标系的原点还在特征点上
			double rot_x = (cos_ori * j - sin_ori * i) / sub_hist_width;
			double rot_y = (sin_ori * j + cos_ori * i) / sub_hist_width;

			//xbin,ybin为落在4*4窗口中像素的下标值，坐标系的原点移动到4*4窗口的左上角的种子点上
			double xbin = rot_x + width / 2 - 0.5;
			double ybin = rot_y + width / 2 - 0.5;

			//仅统计以特征点为中心的5*5正方形里的像素点，描述子本身的窗口是以特征点为中心的4*4正方形
			if (xbin > -1.0 && xbin < width && ybin > -1.0 && ybin < width)
			{
				//计算以特征点为中心，radius为半径圆形区域内所有像素的幅值和梯度方向
				//在旋转前统计幅值和梯度方向
				if (CalcGradMagOri(gauss, x + j, y + i, grad_mag, grad_ori))
				{
					//转换了角度坐标系，并且进行旋转（旋转ori主方向的角度），详细参考1051 ~ 1064行的代码
					grad_ori = (CV_PI - grad_ori) - ori;

					//将所有的角度都归一化到0 ~ 2PI内
					while (grad_ori < 0.0)
						grad_ori += PI2;
					while (grad_ori >= PI2)
						grad_ori -= PI2;

					//将角度0 ~ 2PI转化为数值0 ~ 7
					double obin = grad_ori * (bins / PI2);

					//高斯核以特征点为中心
					double weight = exp(conste * (rot_x * rot_x + rot_y * rot_y));

					//将旋转前计算的幅值赋给旋转后的像素（像素前后只是坐标的变换，相当于做了旋转和放缩，这里认为旋转对单个像素幅值影响）
					InterpHistEntry(hist, xbin, ybin, obin, grad_mag * weight, bins, width);
				}
			}
		}
	}

	return hist;
}

/*
  对特征描绘子进行归一化
  @param feat 特征描绘子
*/
void NormalizeDescr(Keypoint& feat)
{
	double cur, len_inv, len_sq = 0.0;
	int i, d = feat.descr_length;

	//求平方和
	for (i = 0; i < d; i++)
	{
		cur = feat.descriptor[i];
		len_sq += cur * cur;
	}
	len_inv = 1.0 / sqrt(len_sq);

	for (i = 0; i < d; i++)
		feat.descriptor[i] *= len_inv;
}

/*
  将特征描绘子数组转化为特征描绘子矢量
  @param hist 特征描绘子数组
  @param width 邻域划分为4*4的个方格
  @param bins 特征描绘子梯度方向直方图的bins
  @param feature 输出特征描绘子矢量
*/
void HistToDescriptor(double*** hist, int width, int bins, Keypoint& feature)
{
	int int_val, i, r, c, o, k = 0;

	for (r = 0; r < width; r++)
		for (c = 0; c < width; c++)
			for (o = 0; o < bins; o++)
			{
				feature.descriptor[k++] = hist[r][c][o];
			}
	//k = 4 * 4 * 8 = 128
	feature.descr_length = k;

	//截断前进行归一化，去除光照的影响
	NormalizeDescr(feature);

	for (i = 0; i < k; i++)
		//特征描绘子矢量某一维数值超过0.2要进行截断
		if (feature.descriptor[i] > DESCR_MAG_THR)
			feature.descriptor[i] = DESCR_MAG_THR;

	//截断后进行归一化，提高特征描绘子矢量的鉴别性
	NormalizeDescr(feature);

	/* convert floating-point descriptor to integer valued descriptor */
	for (i = 0; i < k; i++)
	{
		int_val = INT_DESCR_FCTR * feature.descriptor[i];
		feature.descriptor[i] = min(255, int_val);
	}
}

/*
  计算特征描绘子，计算特征描绘子直方图，将特征描绘子直方图转化为特征描绘子矢量
  @param hist 特征描绘子数组
  @param width 邻域划分为4*4的个方格
  @param bins 特征描绘子梯度方向直方图的bins
  @param feature 输出特征描绘子矢量
*/
void DescriptorRepresentation(vector<Keypoint>& features, const vector<Mat>& gauss_pyr, int bins, int width)
{
	double*** hist;

	for (int i = 0; i < features.size(); i++)
	{
		hist = CalculateDescrHist(gauss_pyr[features[i].octave * (INTERVALS + 3) + features[i].interval],
			features[i].x, features[i].y, features[i].octave_scale, features[i].ori, bins, width);

		HistToDescriptor(hist, width, bins, features[i]);

		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < width; k++)
			{
				delete[] hist[j][k];
			}
			delete[] hist[j];
		}
		delete[] hist;
	}
}

/*
  比较函数，将特征点按尺度的降序排列
  @param f1 第一个特征点的指针
  @param f2 第二个特征点的指针
  @return 如果f1的尺度小于f2的尺度，返回1；否则返回-1；若相等返回0
*/
bool FeatureCmp(Keypoint& f1, Keypoint& f2)
{
	return f1.scale < f2.scale;
}

//sift 算法
void Sift(const Mat& src, vector<Keypoint>& features, double sigma, int intervals)
{
	Mat init_gray;
	//初始化第0组第0层图像
	CreateInitSmoothGray(src, init_gray, sigma);

	//计算高斯金字塔的组数
	int octaves = log((double)min(init_gray.rows, init_gray.cols)) / log(2.0) - 2;
	std::cout << "rows = " << init_gray.rows << "  cols = " << init_gray.cols << "  octaves = " << octaves << std::endl;
	std::cout << std::endl;

	std::cout << "building gaussian pyramid..." << std::endl;
	std::cout << std::endl;
	vector<Mat> gauss_pyr;
	//生成高斯金字塔
	GaussianPyramid(init_gray, gauss_pyr, octaves, intervals, sigma);

	//write_pyr(gauss_pyr, "gausspyrmaid");
	//display_pyr(gauss_pyr, "gausspyramid");

	std::cout << "building difference of gaussian pyramid..." << std::endl;
	std::cout << std::endl;
	vector<Mat> dog_pyr;
	//生成高斯差分金字塔
	DogPyramid(gauss_pyr, dog_pyr, octaves, intervals);

	//write_pyr(dog_pyr, "dogpyrmaid");
	//display_pyr(dog_pyr, "dogpyramid");

	std::cout << "deatecting local extrema..." << std::endl;
	vector<Keypoint> extrema;
	//检测关键点，并生成关键点vector
	DetectionLocalExtrema(dog_pyr, extrema, octaves, intervals);
	std::cout << "keypoints cout: " << extrema.size() << std::endl;
	std::cout << "extrema detection finished" << std::endl;
	std::cout << "please look dir gausspyramid, dogpyramid and extrema.txt" << std::endl;
	std::cout << std::endl;

	//计算关键点vector中每个关键点相对于第0组第0层的尺度
	CalculateScale(extrema, sigma, intervals);

	//计算关键点vector中每个极值点x、y、scale相对于原图像的x、y、scale
	HalfFeatures(extrema);

	std::cout << "orientation assignment..." << std::endl;
	//分配关键点的主方向
	OrientationAssignment(extrema, features, gauss_pyr);
	std::cout << "features count: " << features.size() << std::endl;
	std::cout << std::endl;

	std::cout << "generating SIFT descriptors..." << std::endl;
	std::cout << std::endl;
	//计算特征描绘子的直方图并生成特征描绘子矢量
	DescriptorRepresentation(features, gauss_pyr, DESCR_HIST_BINS, DESCR_WINDOW_WIDTH);

	//根据特征描绘子矢量的尺度进行排序
	sort(features.begin(), features.end(), FeatureCmp);

	std::cout << "finished......" << std::endl;
	std::cout << std::endl;
}

void Sift_nointerval(const Mat& src, vector<Keypoint>& features, double sigma, int intervals)
{
	Mat init_gray;
	//初始化第0组第0层图像
	CreateInitSmoothGray(src, init_gray, sigma);

	//计算高斯金字塔的组数
	int octaves = log((double)min(init_gray.rows, init_gray.cols)) / log(2.0) - 2;
	std::cout << "rows = " << init_gray.rows << "  cols = " << init_gray.cols << "  octaves = " << octaves << std::endl;
	std::cout << std::endl;

	std::cout << "building gaussian pyramid..." << std::endl;
	std::cout << std::endl;
	vector<Mat> gauss_pyr;
	//生成高斯金字塔
	GaussianPyramid(init_gray, gauss_pyr, octaves, intervals, sigma);

	//write_pyr(gauss_pyr, "gausspyrmaid");
	//display_pyr(gauss_pyr, "gausspyramid");

	std::cout << "building difference of gaussian pyramid..." << std::endl;
	std::cout << std::endl;
	vector<Mat> dog_pyr;
	//生成高斯差分金字塔
	DogPyramid(gauss_pyr, dog_pyr, octaves, intervals);

	//write_pyr(dog_pyr, "dogpyrmaid");
	//display_pyr(dog_pyr, "dogpyramid");

	std::cout << "deatecting local extrema..." << std::endl;
	vector<Keypoint> extrema;
	//检测关键点，并生成关键点vector
	DetectionLocalExtrema_nointerval(dog_pyr, extrema, octaves, intervals);
	std::cout << "keypoints cout: " << extrema.size() << std::endl;
	std::cout << "extrema detection finished" << std::endl;
	std::cout << "please look dir gausspyramid, dogpyramid and extrema.txt" << std::endl;
	std::cout << std::endl;

	//计算关键点vector中每个关键点相对于第0组第0层的尺度
	CalculateScale(extrema, sigma, intervals);

	//计算关键点vector中每个极值点x、y、scale相对于原图像的x、y、scale
	HalfFeatures(extrema);

	std::cout << "orientation assignment..." << std::endl;
	//分配关键点的主方向
	OrientationAssignment(extrema, features, gauss_pyr);
	std::cout << "features count: " << features.size() << std::endl;
	std::cout << std::endl;

	std::cout << "generating SIFT descriptors..." << std::endl;
	std::cout << std::endl;
	//计算特征描绘子的直方图并生成特征描绘子矢量
	DescriptorRepresentation(features, gauss_pyr, DESCR_HIST_BINS, DESCR_WINDOW_WIDTH);

	//根据特征描绘子矢量的尺度进行排序
	sort(features.begin(), features.end(), FeatureCmp);

	std::cout << "finished......" << std::endl;
	std::cout << std::endl;
}

void DrawSiftFeature(Mat& src, Keypoint& feat, cv::Scalar color)
{
	int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
	double scl, ori;
	double scale = 5.0;
	double hscale = 0.75;
	cv::Point start, end, h1, h2;

	/* compute points for an arrow scaled and rotated by feat's scl and ori */
	start_x = cvRound(feat.dx);
	start_y = cvRound(feat.dy);
	scl = feat.scale;
	ori = feat.ori;
	len = cvRound(scl * scale);
	hlen = cvRound(scl * hscale);
	blen = len - hlen;
	end_x = cvRound(len * cos(ori)) + start_x;
	end_y = cvRound(len * -sin(ori)) + start_y;

	h1_x = cvRound(blen * cos(ori + CV_PI / 18.0)) + start_x;
	h1_y = cvRound(blen * -sin(ori + CV_PI / 18.0)) + start_y;
	h2_x = cvRound(blen * cos(ori - CV_PI / 18.0)) + start_x;
	h2_y = cvRound(blen * -sin(ori - CV_PI / 18.0)) + start_y;

	start = cv::Point(start_x, start_y);
	end = cv::Point(end_x, end_y);
	h1 = cv::Point(h1_x, h1_y);
	h2 = cv::Point(h2_x, h2_y);

	//所以最后线的长短代表着尺度的大小
	line(src, start, end, color, 1, 8, 0);
	line(src, end, h1, color, 1, 8, 0);
	line(src, end, h2, color, 1, 8, 0);
	//circle(src, Point(start_x, start_y), blen, color, 1);
}

void DrawSiftFeatures(Mat& src, vector<Keypoint>& features)
{
	cv::Scalar color = CV_RGB(0, 255, 0);
	for (int i = 0; i < features.size(); i++)
	{
		DrawSiftFeature(src, features[i], color);
	}
}

void DrawKeyPoints(Mat& src, vector<Keypoint>& keypoints)
{
	int j = 0;
	for (int i = 0; i < keypoints.size(); i++)
	{
		cv::Scalar color = CV_RGB(0, 0, 255);
		circle(src, Point(keypoints[i].dx, keypoints[i].dy), 3, color, 1);
		j++;
	}
}

void DrawKeyPointsRight(Mat& src, vector<Keypoint>& keypoints, cv::Size size)
{
	int j = 0;
	for (int i = 0; i < keypoints.size(); i++)
	{
		cv::Scalar color = CV_RGB(0, 0, 255);
		//左右拼接使用
		circle(src, Point(keypoints[i].dx + size.width, keypoints[i].dy), 3, color);
		j++;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

void cv64f_to_cv8U(const Mat& src, Mat& dst)
{
	normalize(src, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8U);
}

const char* GetFileName(const char* dir, int i)
{
	char* name = new char[50];
	std::cout << "name: " << dir << " " << i << std::endl;
	//sprintf(name, "%s\%d.jpg", dir, i);
	return name;
}

//通过转换后保存的图像，会失真,和imshow显示出的图像相差很大
void writecv64f(const char* filename, const Mat& mat)
{
	Mat dst;
	cv64f_to_cv8U(mat, dst);
	imwrite(filename, dst);
}

void write_pyr(const vector<Mat>& pyr, const char* dir)
{
	for (int i = 0; i < pyr.size(); i++)
	{
		writecv64f(GetFileName(dir, i), pyr[i]);
	}
}

void display_pyr(const vector<Mat>& pyr, string dir)
{
	for (int i = 0; i < pyr.size(); i++)
	{
		std::cout << "name: " << dir << " " << i << std::endl;
		string name = dir + to_string(i) + ".jpg";
		std::cout << name << std::endl;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void read_features(vector<Keypoint>& features, const char* file)
{
	//文件读操作，存储设备读取到内存中
	ifstream in(file);
	int n = 0, dims = 0;
	//用析取器(>>)从文件中读入数据
	in >> n >> dims;
	std::cout << n << " " << dims << std::endl;
	for (int i = 0; i < n; i++)
	{
		Keypoint key;
		//读入特征点的x、y坐标，尺度scale，方向ori
		in >> key.dy >> key.dx >> key.scale >> key.ori;

		//读入特征描绘子各维度的数据
		for (int j = 0; j < dims; j++)
		{
			in >> key.descriptor[j];
		}
		features.push_back(key);
	}
	in.close();
}

void write_features(const vector<Keypoint>& features, const char* file)
{
	//文件写操作，内存写入存储设备
	ofstream dout(file);
	//用插入器(<<)向文件中写入数据
	dout << features.size() << " " << FEATURE_ELEMENT_LENGTH << endl;
	for (int i = 0; i < features.size(); i++)
	{
		//写入特征点的x、y坐标，尺度scale，方向ori
		dout << features[i].dy << " " << features[i].dx << " " << features[i].scale << " " << features[i].ori << endl;
		for (int j = 0; j < FEATURE_ELEMENT_LENGTH; j++)
		{
			//每行数据到二十个则换行输出
			if (j % 20 == 0)
				dout << endl;
			dout << features[i].descriptor[j] << " ";
		}
		dout << endl;
		dout << endl;
	}
	dout.close();
}

void testInverse3D()
{
	double H[9] = { 3,2,3, 4, 5, 6, 7, 8, 9 };
	double H_inve[9] = { 0 };
	double r[9] = { 0 };

	if (!Inverse3D(H, H_inve))
	{
		cout << "Not inverse." << endl;
		return;
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			r[3 * i + j] = 0;
			for (int k = 0; k < 3; k++)
			{
				r[3 * i + j] += H[3 * i + k] * H_inve[3 * k + j];
			}
			cout << r[3 * i + j] << " ";
		}
		cout << endl;
	}
}