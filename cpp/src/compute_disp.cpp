#include "iostream"
#include "opencv2/opencv.hpp"
#include <sister/SisterMultiviewDisparities.hpp>

using namespace cv;

int main(int argc, char **argv)
{
	// Check input params
	if (argc != 3)
	{
		std::cout << "expected <input folder> <dmax>" << std::endl;
		return -1;
	}

	char *inputFolder = argv[1];
	int dispCount = atoi(argv[2]);

	cv::Mat center = cv::imread(string(inputFolder) + "center.png");
	cv::Mat right = cv::imread(string(inputFolder) + "right.png");
	cv::Mat top = cv::imread(string(inputFolder) + "top.png");
	cv::Mat left = cv::imread(string(inputFolder) + "left.png");
	cv::Mat bottom = cv::imread(string(inputFolder) + "bottom.png");

	// Multiview matcher
	SisterMultiviewDisparities multiview(
		center,
		right,
		top,
		left,
		bottom);

	// Compute disparities
	cv::Mat result_horizontal, result_vertical, result_multiview;
	multiview.compute_disparities(dispCount, result_multiview, result_horizontal, result_vertical);

	// Colorize disparities
	cv::normalize(result_horizontal, result_horizontal, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::normalize(result_vertical, result_vertical, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::normalize(result_multiview, result_multiview, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::applyColorMap(result_horizontal, result_horizontal, cv::COLORMAP_MAGMA);
	cv::applyColorMap(result_vertical, result_vertical, cv::COLORMAP_MAGMA);
	cv::applyColorMap(result_multiview, result_multiview, cv::COLORMAP_MAGMA);

	// Blend RGB image and multiview disparity
	cv::Mat blended;
	float alpha_blending_rgb = 0.1;
	cv::addWeighted(center, alpha_blending_rgb, result_multiview, 1.0 - alpha_blending_rgb, 0.0, blended);

	// View
	cv::imshow("image", center);
	cv::imshow("blended", blended);
	cv::imshow("disp_horizontal", result_horizontal);
	cv::imshow("disp_vertical", result_horizontal);
	cv::imshow("disp_multiview", result_multiview);
	cv::waitKey(0);

	return 0;
}
