#include "iostream"
#include "stereoalgo.h"
#include "opencv2/opencv.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include <ctime>
#include "DSI.h"

#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace cv;

void tile(const vector<Mat> &src, Mat &dst, int grid_x, int grid_y)
{
	// patch size
	int width = dst.cols / grid_x;
	int height = dst.rows / grid_y;
	printf("%d\n", height);
	printf("%d\n", width);
	// iterate through grid
	int k = 0;
	for (int i = 0; i < grid_y; i++)
	{
		for (int j = 0; j < grid_x; j++)
		{
			Mat s = src[k++];
			//resize(s,s,Size(width,height));
			s.copyTo(dst(Rect(j * width, i * height, width, height)));
		}
	}
}

_DSI doMultiStereo(uint8 *imC_data, uint8 *im0_data, uint8 *imCto90_data, uint8 *im90_data, uint8 *imCto180_data, uint8 *im180_data, uint8 *imCto270_data, uint8 *im270_data, float32 *outputL, float32 *outputR, int width, int height, int method, uint16 paths, const int numThreads, const int numStrips, const int dispCount, const int mode, const int mccnn)
{
	// Maximum disparity encoded
	const int maxDisp = dispCount - 1;
	Mat outputL_HW = Mat(height, width, CV_32FC1);
	float32 *ptrL_hw = (float32 *)outputL_HW.data;
	Mat outputR_HW = Mat(height, width, CV_32FC1);
	float32 *ptrR_hw = (float32 *)outputR_HW.data;

	Mat outputL_WH = Mat(width, height, CV_32FC1);
	float32 *ptrL_wh = (float32 *)outputL_WH.data;
	Mat outputR_WH = Mat(width, height, CV_32FC1);
	float32 *ptrR_wh = (float32 *)outputR_WH.data;

	// Alloc dsi
	uint16 *dsiC0 = (uint16 *)_mm_malloc(width * height * dispCount * sizeof(uint16), 16);
	uint16 *dsiC90 = (uint16 *)_mm_malloc(width * height * dispCount * sizeof(uint16), 16);
	uint16 *dsiC180 = (uint16 *)_mm_malloc(width * height * dispCount * sizeof(uint16), 16);
	uint16 *dsiC270 = (uint16 *)_mm_malloc(width * height * dispCount * sizeof(uint16), 16);
	uint16 *dsiC90f = (uint16 *)_mm_malloc(width * height * dispCount * sizeof(uint16), 16);
	uint16 *dsiC180f = (uint16 *)_mm_malloc(width * height * dispCount * sizeof(uint16), 16);
	uint16 *dsiC270f = (uint16 *)_mm_malloc(width * height * dispCount * sizeof(uint16), 16);

	uint16 *multidsi = (uint16 *)_mm_malloc(width * height * dispCount * sizeof(uint16), 16);

	uint16 *sum_dsi = (uint16 *)_mm_malloc(width * height * dispCount * sizeof(uint16), 16);

	// 1) Matching cost
	if (mccnn == 0)
	{
		// 1) AD-Census
		ad_census(imC_data, im0_data, height, width, dispCount, dsiC0, numThreads);
		ad_census(imCto90_data, im90_data, width, height, dispCount, dsiC90f, numThreads);
		ad_census(imCto180_data, im180_data, height, width, dispCount, dsiC180f, numThreads);
		ad_census(imCto270_data, im270_data, width, height, dispCount, dsiC270f, numThreads);
	}
	else
	{
		// 1) MC-CNN
		float *temp;
		int fd = open("../left.bin", O_RDONLY);
		temp = (float *)mmap(NULL, dispCount * height * width * sizeof(float), PROT_READ, MAP_SHARED, fd, 0);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				for (int k = 0; k < dispCount; k++)
					dsiC0[i * width * dispCount + j * dispCount + k] = temp[k * width * height + i * width + j];
		free(temp);
		//	  	close(fd);
	}

	// 1.1) LRC to obtain confidences
	Mat lrc0 = Mat(height, width, CV_8UC1);
	Mat lrc90 = Mat(width, height, CV_8UC1);
	Mat lrc180 = Mat(height, width, CV_8UC1);
	Mat lrc270 = Mat(width, height, CV_8UC1);

	if (mode != 2)
	{
		// Center to left
		WTALeft_SSE(ptrL_hw, dsiC0, width, height, maxDisp, 1);
		WTARight_SSE(ptrR_hw, dsiC0, width, height, maxDisp, 1);
		median3x3_SSE(ptrL_hw, ptrL_hw, width, height);
		median3x3_SSE(ptrR_hw, ptrR_hw, width, height);
		doLRCheck(ptrL_hw, ptrR_hw, width, height, 5);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (ptrL_hw[i * width + j] <= 0 || j < dispCount)
					lrc0.at<uchar>(i, j) = 0;
				else
					lrc0.at<uchar>(i, j) = 1;
		// Center to right
		WTALeft_SSE(ptrL_hw, dsiC180f, width, height, maxDisp, 1);
		WTARight_SSE(ptrR_hw, dsiC180f, width, height, maxDisp, 1);
		median3x3_SSE(ptrL_hw, ptrL_hw, width, height);
		median3x3_SSE(ptrR_hw, ptrR_hw, width, height);
		doLRCheck(ptrL_hw, ptrR_hw, width, height, 5);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (ptrL_hw[i * width + j] <= 0 || j < dispCount)
					lrc180.at<uchar>(i, width - 1 - j) = 0;
				else
					lrc180.at<uchar>(i, width - 1 - j) = 1;
	}

	if (mode != 1)
	{
		// Center to top
		WTALeft_SSE(ptrL_wh, dsiC90f, height, width, maxDisp, 1);
		WTARight_SSE(ptrR_wh, dsiC90f, height, width, maxDisp, 1);
		median3x3_SSE(ptrL_wh, ptrL_wh, height, width);
		median3x3_SSE(ptrR_wh, ptrR_wh, height, width);
		doLRCheck(ptrL_wh, ptrR_wh, height, width, 5);
		for (int i = 0; i < width; i++)
			for (int j = 0; j < height; j++)
				if (ptrL_wh[i * height + j] <= 0 || j < dispCount)
					lrc90.at<uchar>(i, j) = 0;
				else
					lrc90.at<uchar>(i, j) = 1;
		transpose(lrc90, lrc90);
		flip(lrc90, lrc90, -1);

		// Center to bottom
		WTALeft_SSE(ptrL_wh, dsiC270f, height, width, maxDisp, 1);
		WTARight_SSE(ptrR_wh, dsiC270f, height, width, maxDisp, 1);
		median3x3_SSE(ptrL_wh, ptrL_wh, height, width);
		median3x3_SSE(ptrR_wh, ptrR_wh, height, width);
		doLRCheck(ptrL_wh, ptrR_wh, height, width, 5);
		for (int i = 0; i < width; i++)
			for (int j = 0; j < height; j++)
				if (ptrL_wh[i * height + j] <= 0 || j < dispCount)
					lrc270.at<uchar>(i, j) = 0;
				else
					lrc270.at<uchar>(i, j) = 1;
		transpose(lrc270, lrc270);
		flip(lrc270, lrc270, 1);
	}

	// 1.2) Transpose DSIs and SUM (weighted by confidence)
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			for (int d = 0; d < dispCount; d++)
			{
				dsiC180[i * width * dispCount + j * dispCount + d] = dsiC180f[i * width * dispCount + (width - 1 - j) * dispCount + d];
				dsiC90[i * width * dispCount + j * dispCount + d] = dsiC90f[(width - 1 - j) * height * dispCount + (height - 1 - i) * dispCount + d];
				dsiC270[i * width * dispCount + j * dispCount + d] = dsiC270f[(width - 1 - j) * height * dispCount + (i)*dispCount + d];
				switch (mode)
				{
				case 0:
					multidsi[i * width * dispCount + j * dispCount + d] = lrc0.at<uchar>(i, j) * dsiC0[i * width * dispCount + j * dispCount + d] + lrc180.at<uchar>(i, j) * dsiC180[i * width * dispCount + j * dispCount + d] + lrc90.at<uchar>(i, j) * dsiC90[i * width * dispCount + j * dispCount + d] + lrc270.at<uchar>(i, j) * dsiC270[i * width * dispCount + j * dispCount + d];
					//multidsi[i * width * dispCount + j * dispCount + d] = dsiC0[i * width * dispCount + j * dispCount + d] + dsiC180[i * width * dispCount + j * dispCount + d] + dsiC90[i * width * dispCount + j * dispCount + d] + dsiC270[i * width * dispCount + j * dispCount + d];
					break;
				case 1:
					multidsi[i * width * dispCount + j * dispCount + d] = lrc0.at<uchar>(i, j) * dsiC0[i * width * dispCount + j * dispCount + d] + lrc180.at<uchar>(i, j) * dsiC180[i * width * dispCount + j * dispCount + d];
					//multidsi[i * width * dispCount + j * dispCount + d] = dsiC0[i * width * dispCount + j * dispCount + d] + dsiC180[i * width * dispCount + j * dispCount + d];
					break;
				case 2:
					multidsi[i * width * dispCount + j * dispCount + d] = lrc90.at<uchar>(i, j) * dsiC90[i * width * dispCount + j * dispCount + d] + lrc270.at<uchar>(i, j) * dsiC270[i * width * dispCount + j * dispCount + d];
					//multidsi[i * width * dispCount + j * dispCount + d] = dsiC90[i * width * dispCount + j * dispCount + d] + dsiC270[i * width * dispCount + j * dispCount + d];
					break;
				}
			}

	// 2) multi-view SGM
	sgm(imC_data, height, width, dispCount, multidsi, sum_dsi, 7, 17, 8);

	// 3) WTA
	WTALeft_SSE(outputL, sum_dsi, width, height, maxDisp, 1);

	// create the DSI for confidence
	_DSI volume = DSI_init(height, width, 0, dispCount, 0);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			for (int d = 0; d < dispCount; d++)
			{
				volume.values[d].ptr<float>(i)[j] = sum_dsi[width * dispCount * i + j * dispCount + d];
			}

	// Free dsi
	_mm_free(dsiC0);
	_mm_free(dsiC90);
	_mm_free(dsiC180);
	_mm_free(dsiC270);
	_mm_free(dsiC90f);
	_mm_free(dsiC180f);
	_mm_free(dsiC270f);
	_mm_free(multidsi);
	_mm_free(sum_dsi);
	return volume;
}

_DSI doStereo(uint8 *imC_data, uint8 *im0_data, float32 *outputL, float32 *outputR, int width, int height, int method, uint16 paths, const int numThreads, const int numStrips, const int dispCount)
{
	// Maximum disparity encoded
	const int maxDisp = dispCount - 1;

	// Alloc dsi
	uint16 *dsi = (uint16 *)_mm_malloc(width * height * dispCount * sizeof(uint16), 16);
	uint16 *sum_dsi = (uint16 *)_mm_malloc(width * height * dispCount * sizeof(uint16), 16);
	printf("HERE\n");
	// 1) AD-Census matching cost
	ad_census(imC_data, im0_data, height, width, dispCount, dsi, numThreads);

	// 2) SGM
	sgm(imC_data, height, width, dispCount, dsi, sum_dsi, 7, 17, 8);

	// 3) WTA
	WTALeft_SSE(outputL, sum_dsi, width, height, maxDisp, 1);
	WTARight_SSE(outputR, sum_dsi, width, height, maxDisp, 1);
	median3x3_SSE(outputL, outputL, width, height);
	median3x3_SSE(outputR, outputR, width, height);

	// 4) LRC
	doLRCheck(outputL, outputR, width, height, 5);

	// create the DSI for confidence
	_DSI volume = DSI_init(height, width, 0, dispCount, 0);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			for (int d = 0; d < dispCount; d++)
			{
				volume.values[d].ptr<float>(i)[j] = sum_dsi[width * dispCount * i + j * dispCount + d];
			}

	// Free dsi
	_mm_free(dsi);
	_mm_free(sum_dsi);
	return volume;
}

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


	for (int i = 0; i < 1; i++)
	{
		Mat bufferC, buffer0, buffer90, buffer180, buffer270, imC, im0, im90, im180, im270;

		bufferC = cv::imread(string(inputFolder) + "center.png");
		buffer0 = cv::imread(string(inputFolder) + "right.png");
		buffer90 = cv::imread(string(inputFolder) + "top.png");
		buffer180 = cv::imread(string(inputFolder) + "left.png");
		buffer270 = cv::imread(string(inputFolder) + "bottom.png");
	

		std::cout << "SHAPE "<< bufferC.size()<<std::endl;

		cvtColor(bufferC, imC, cv::COLOR_BGR2GRAY);
		cvtColor(buffer0, im0, cv::COLOR_BGR2GRAY);
		cvtColor(buffer90, im90, cv::COLOR_BGR2GRAY);
		cvtColor(buffer180, im180, cv::COLOR_BGR2GRAY);
		cvtColor(buffer270, im270, cv::COLOR_BGR2GRAY);

		copyMakeBorder(imC, imC, dispCount, dispCount, dispCount, dispCount, BORDER_REPLICATE, 0);
		copyMakeBorder(im0, im0, dispCount, dispCount, dispCount, dispCount, BORDER_REPLICATE, 0);
		copyMakeBorder(im90, im90, dispCount, dispCount, dispCount, dispCount, BORDER_REPLICATE, 0);
		copyMakeBorder(im180, im180, dispCount, dispCount, dispCount, dispCount, BORDER_REPLICATE, 0);
		copyMakeBorder(im270, im270, dispCount, dispCount, dispCount, dispCount, BORDER_REPLICATE, 0);

		int width = imC.cols, height = imC.rows;
		std::cout << "Frame " << i << ": " << width << "x" << height << std::endl;

		Mat dispC0 = Mat(height, width, CV_32FC1);
		Mat dispC90f = Mat(width, height, CV_32FC1);
		Mat dispC180 = Mat(height, width, CV_32FC1);
		Mat dispC270f = Mat(width, height, CV_32FC1);
		Mat dispRight = Mat(height, width, CV_32FC1);

		Mat horizontalDisp = Mat(height, width, CV_32FC1);
		Mat verticalDisp = Mat(height, width, CV_32FC1);
		Mat multiDisp = Mat(height, width, CV_32FC1);
		Mat consensus = Mat(height, width, CV_32FC1);

		// Flipping images
		// CENTER -> RIGHT
		Mat imCf_to180, im180f;
		flip(imC, imCf_to180, 1);
		flip(im180, im180f, 1);
		// CENTER -> TOP
		Mat imCf_to90, im90f;
		transpose(imC, imCf_to90);
		flip(imCf_to90, imCf_to90, -1);
		transpose(im90, im90f);
		flip(im90f, im90f, -1);
		// CENTER -> BOTTOM
		Mat imCf_to270, im270f;
		transpose(imC, imCf_to270);
		flip(imCf_to270, imCf_to270, 0);
		transpose(im270, im270f);
		flip(im270f, im270f, 0);

		// Run stereo!
		clock_t begin, end;


		// Multiview!!!
		begin = clock();
		doMultiStereo((uint8 *)(imC.data), (uint8 *)(im0.data), (uint8 *)(imCf_to90.data), (uint8 *)(im90f.data), (uint8 *)(imCf_to180.data), (uint8 *)(im180f.data), (uint8 *)(imCf_to270.data), (uint8 *)(im270f.data), (float32 *)(multiDisp.data), (float32 *)(dispRight.data), width, height, 1, 8, 4, 4, dispCount, 0, 0);
		end = clock();
		std::cout << "MV Disparity processed in " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl;

		begin = clock();
		doMultiStereo((uint8 *)(imC.data), (uint8 *)(im0.data), (uint8 *)(imCf_to90.data), (uint8 *)(im90f.data), (uint8 *)(imCf_to180.data), (uint8 *)(im180f.data), (uint8 *)(imCf_to270.data), (uint8 *)(im270f.data), (float32 *)(horizontalDisp.data), (float32 *)(dispRight.data), width, height, 1, 8, 4, 4, dispCount, 1, 0);
		end = clock();
		std::cout << "Horizontal Disparity processed in " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl;

		begin = clock();
		doMultiStereo((uint8 *)(imC.data), (uint8 *)(im0.data), (uint8 *)(imCf_to90.data), (uint8 *)(im90f.data), (uint8 *)(imCf_to180.data), (uint8 *)(im180f.data), (uint8 *)(imCf_to270.data), (uint8 *)(im270f.data), (float32 *)(verticalDisp.data), (float32 *)(dispRight.data), width, height, 1, 8, 4, 4, dispCount, 2, 0);
		end = clock();
		std::cout << "Vertical Disparity processed in " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl;

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				if (fabs(dispC0.at<float>(i, j) - dispC90f.at<float>(i, j)) < 3 && fabs(dispC90f.at<float>(i, j) - dispC180.at<float>(i, j)) < 3 && fabs(dispC180.at<float>(i, j) - dispC270f.at<float>(i, j)) < 3)
					consensus.at<float>(i, j) = (dispC0.at<float>(i, j) + dispC90f.at<float>(i, j) + dispC180.at<float>(i, j) + dispC270f.at<float>(i, j)) / 4;
				else
				{
					int maxDisp = 0;
					if (dispC0.at<float>(i, j) > maxDisp)
						maxDisp = dispC0.at<float>(i, j);
					if (dispC90f.at<float>(i, j) > maxDisp)
						maxDisp = dispC90f.at<float>(i, j);
					if (dispC180.at<float>(i, j) > maxDisp)
						maxDisp = dispC180.at<float>(i, j);
					if (dispC270f.at<float>(i, j) > maxDisp)
						maxDisp = dispC270f.at<float>(i, j);
					consensus.at<float>(i, j) = maxDisp; //(dispC0.at<float>(i,j)+dispC90f.at<float>(i,j)+dispC180.at<float>(i,j)+dispC270f.at<float>(i,j))/nonzeros;
				}
			}


		horizontalDisp.convertTo(horizontalDisp, CV_16UC1);
		verticalDisp.convertTo(verticalDisp, CV_16UC1);
		multiDisp.convertTo(multiDisp, CV_16UC1);


		// Output display
		cv::Mat result_horizontal = horizontalDisp(Rect(dispCount, dispCount, width - (dispCount * 2), height - (dispCount * 2))) * 255;
		cv::Mat result_vertical = verticalDisp(Rect(dispCount, dispCount, width - (dispCount * 2), height - (dispCount * 2))) * 255;
		cv::Mat result_multiview = multiDisp(Rect(dispCount, dispCount, width - (dispCount * 2), height - (dispCount * 2))) * 255;

		cv::normalize(result_horizontal, result_horizontal, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::normalize(result_vertical, result_vertical, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::normalize(result_multiview, result_multiview, 0, 255, cv::NORM_MINMAX, CV_8UC1);

		cv::applyColorMap(result_horizontal, result_horizontal, cv::COLORMAP_MAGMA);
		cv::applyColorMap(result_vertical, result_vertical, cv::COLORMAP_MAGMA);
		cv::applyColorMap(result_multiview, result_multiview, cv::COLORMAP_MAGMA);

		cv::Mat blended;
		float alpha_blending_rgb = 0.1;
		cv::addWeighted(bufferC,alpha_blending_rgb, result_multiview, 1.0 - alpha_blending_rgb, 0.0, blended);

		cv::imshow("image", bufferC);
		cv::imshow("blended", blended);
		cv::imshow("disp_horizontal", result_horizontal);
		cv::imshow("disp_vertical", result_horizontal);
		cv::imshow("disp_multiview", result_multiview);
		cv::waitKey(0);
		
	}

	return 0;
}