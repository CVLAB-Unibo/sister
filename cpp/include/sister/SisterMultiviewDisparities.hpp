#ifndef SisterMultiviewDisparities_H
#define SisterMultiviewDisparities_H

#include "iostream"
#include "stereoalgo.h"
#include "opencv2/opencv.hpp"
#include <ctime>

#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace cv;
using namespace std;

class SisterMultiviewDisparities
{

public:
    SisterMultiviewDisparities(cv::Mat center, cv::Mat right, cv::Mat top, cv::Mat left, cv::Mat bottom) : center(center), right(right), top(top), left(left), bottom(bottom)
    {
    }

    void compute_disparities(int dispCount, cv::Mat &disp_multiview, cv::Mat &disp_horizontal, cv::Mat &disp_vertical)
    {
        cv::Mat imC, im0, im90, im180, im270;
        cvtColor(this->center, imC, cv::COLOR_BGR2GRAY);
        cvtColor(this->right, im0, cv::COLOR_BGR2GRAY);
        cvtColor(this->top, im90, cv::COLOR_BGR2GRAY);
        cvtColor(this->left, im180, cv::COLOR_BGR2GRAY);
        cvtColor(this->bottom, im270, cv::COLOR_BGR2GRAY);

        copyMakeBorder(imC, imC, dispCount, dispCount, dispCount, dispCount, BORDER_REPLICATE, 0);
        copyMakeBorder(im0, im0, dispCount, dispCount, dispCount, dispCount, BORDER_REPLICATE, 0);
        copyMakeBorder(im90, im90, dispCount, dispCount, dispCount, dispCount, BORDER_REPLICATE, 0);
        copyMakeBorder(im180, im180, dispCount, dispCount, dispCount, dispCount, BORDER_REPLICATE, 0);
        copyMakeBorder(im270, im270, dispCount, dispCount, dispCount, dispCount, BORDER_REPLICATE, 0);

        int width = imC.cols, height = imC.rows;

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
        this->doMultiStereo((uint8 *)(imC.data), (uint8 *)(im0.data), (uint8 *)(imCf_to90.data), (uint8 *)(im90f.data), (uint8 *)(imCf_to180.data), (uint8 *)(im180f.data), (uint8 *)(imCf_to270.data), (uint8 *)(im270f.data), (float32 *)(multiDisp.data), (float32 *)(dispRight.data), width, height, 1, 8, 4, 4, dispCount, 0);
        end = clock();
        std::cout << "MV Disparity processed in " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl;

        begin = clock();
        this->doMultiStereo((uint8 *)(imC.data), (uint8 *)(im0.data), (uint8 *)(imCf_to90.data), (uint8 *)(im90f.data), (uint8 *)(imCf_to180.data), (uint8 *)(im180f.data), (uint8 *)(imCf_to270.data), (uint8 *)(im270f.data), (float32 *)(horizontalDisp.data), (float32 *)(dispRight.data), width, height, 1, 8, 4, 4, dispCount, 1);
        end = clock();
        std::cout << "Horizontal Disparity processed in " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl;

        begin = clock();
        this->doMultiStereo((uint8 *)(imC.data), (uint8 *)(im0.data), (uint8 *)(imCf_to90.data), (uint8 *)(im90f.data), (uint8 *)(imCf_to180.data), (uint8 *)(im180f.data), (uint8 *)(imCf_to270.data), (uint8 *)(im270f.data), (float32 *)(verticalDisp.data), (float32 *)(dispRight.data), width, height, 1, 8, 4, 4, dispCount, 2);
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
        disp_horizontal = horizontalDisp(Rect(dispCount, dispCount, width - (dispCount * 2), height - (dispCount * 2))) * 255;
        disp_vertical = verticalDisp(Rect(dispCount, dispCount, width - (dispCount * 2), height - (dispCount * 2))) * 255;
        disp_multiview = multiDisp(Rect(dispCount, dispCount, width - (dispCount * 2), height - (dispCount * 2))) * 255;
    }

protected:
    void doStereo(uint8 *imC_data, uint8 *im0_data, float32 *outputL, float32 *outputR, int width, int height, int method, uint16 paths, const int numThreads, const int numStrips, const int dispCount)
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

       
        // Free dsi
        _mm_free(dsi);
        _mm_free(sum_dsi);
    }

    void doMultiStereo(uint8 *imC_data, uint8 *im0_data, uint8 *imCto90_data, uint8 *im90_data, uint8 *imCto180_data, uint8 *im180_data, uint8 *imCto270_data, uint8 *im270_data, float32 *outputL, float32 *outputR, int width, int height, int method, uint16 paths, const int numThreads, const int numStrips, const int dispCount, const int mode)
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

        
        // 1) AD-Census
        ad_census(imC_data, im0_data, height, width, dispCount, dsiC0, numThreads);
        ad_census(imCto90_data, im90_data, width, height, dispCount, dsiC90f, numThreads);
        ad_census(imCto180_data, im180_data, height, width, dispCount, dsiC180f, numThreads);
        ad_census(imCto270_data, im270_data, width, height, dispCount, dsiC270f, numThreads);
        

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
    }

    cv::Mat center, right, top, left, bottom;
};

#endif //SisterMultiviewDisparities_H