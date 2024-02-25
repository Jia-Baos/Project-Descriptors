#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#ifdef _DEBUG
#pragma comment(lib, "D:/opencv_contrib/build/x64/vc17/lib/opencv_world460d.lib")
#else
#pragma comment(lib, "D:/opencv_contrib/build/x64/vc17/lib/opencv_world460.lib")
#endif // _DEBUG

//ãÐÖµ´¦Àí+RANSAC
void AKazeRegistration(const cv::Mat src1, const cv::Mat src2,
	std::vector<cv::Point2f>& src1_keypoints_ransac_inliers,
	std::vector<cv::Point2f>& src2_keypoints_ransac_inliers)
{
	cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();

	std::vector<cv::KeyPoint> src1_keypoints, src2_keypoints;

	//-- Computing keypoints
	detector->detect(src1, src1_keypoints);
	detector->detect(src2, src2_keypoints);

	//-- Compute descriptors
	cv::Mat src1_descriptors, src2_descriptors;
	detector->detectAndCompute(src1, cv::Mat(),
		src1_keypoints, src1_descriptors);
	detector->detectAndCompute(src2, cv::Mat(),
		src2_keypoints, src2_descriptors);

	//-- Matching descriptor vectors with a FLANN based matcher
	if (src1_descriptors.type() != CV_32F || src2_descriptors.type() != CV_32F)
	{
		src1_descriptors.convertTo(src1_descriptors, CV_32F);
		src2_descriptors.convertTo(src2_descriptors, CV_32F);
	}

	cv::Ptr<cv::DescriptorMatcher> matcher =
		cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	std::vector<std::vector<cv::DMatch>> knn_matches;
	matcher->knnMatch(src1_descriptors, src2_descriptors, knn_matches, 2);

	//--filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.6f;
	std::vector<cv::DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance <
			ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.emplace_back(knn_matches[i][0]);
		}
	}

	//-- Obtain the good matches
	std::vector<cv::DMatch> inliers;
	std::vector<cv::Point2f> src1_keypoints_ransac;
	std::vector<cv::Point2f> src2_keypoints_ransac;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		src1_keypoints_ransac.emplace_back(src1_keypoints[good_matches[i].queryIdx].pt);
		src2_keypoints_ransac.emplace_back(src2_keypoints[good_matches[i].trainIdx].pt);
	}

	//-- RANSAC FindFundamental to obtain more precise matches
	std::vector<uchar> ransac_status;
	cv::findFundamentalMat(src1_keypoints_ransac, src2_keypoints_ransac,
		ransac_status, cv::FM_RANSAC);

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		if (ransac_status[i] != 0)
		{
			inliers.emplace_back(good_matches[i]);
		}
	}

	for (size_t i = 0; i < inliers.size(); i++)
	{
		src1_keypoints_ransac_inliers.emplace_back(src1_keypoints[inliers[i].queryIdx].pt);
		src2_keypoints_ransac_inliers.emplace_back(src2_keypoints[inliers[i].trainIdx].pt);
	}
}

int main(int argc, char* argv[])
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
	std::cout << "Version: " << CV_VERSION << std::endl;

	const std::string fixed_image_path = "E:/paper2-dataset/template.png";
	const std::string moved_image_path = "E:/paper2-dataset/tested.png";
	std::string target_path = "E:/paper2-dataset/tested-res.png";

	cv::Mat fixed_image = cv::imread(fixed_image_path);
	cv::Mat moved_image = cv::imread(moved_image_path);

	std::chrono::steady_clock::time_point in_time =
		std::chrono::steady_clock::now();

	cv::resize(moved_image, moved_image, fixed_image.size());

	cv::cvtColor(fixed_image, fixed_image, cv::COLOR_BGR2GRAY);
	cv::cvtColor(moved_image, moved_image, cv::COLOR_BGR2GRAY);

	cv::Mat moved_image_warpped;
	std::vector<cv::Point2f> src1_inliers, src2_inliers;
	AKazeRegistration(fixed_image, moved_image, src1_inliers, src2_inliers);
	cv::Mat homography = cv::findHomography(src2_inliers, src1_inliers, cv::RANSAC);
	cv::warpPerspective(moved_image, moved_image_warpped, homography, fixed_image.size());

	std::chrono::steady_clock::time_point out_time =
		std::chrono::steady_clock::now();

	double spend_time =
		std::chrono::duration<double>(out_time - in_time).count();
	std::cout << "spend time: " << spend_time << std::endl;

	cv::namedWindow("pure residual map", cv::WINDOW_NORMAL);
	cv::imshow("pure residual map", cv::abs(fixed_image - moved_image));

	cv::namedWindow("residual map", cv::WINDOW_NORMAL);
	cv::imshow("residual map", cv::abs(fixed_image - moved_image_warpped));

	cv::waitKey();
	return 0;
}