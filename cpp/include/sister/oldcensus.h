#ifndef _CENSUS
#define _CENSUS
#define _CRT_SECURE_NO_WARNINGS

#include<stdio.h>
#include<stdlib.h>

#include<opencv/cv.h>
#include<opencv/highgui.h>

using namespace cv;

void census_transform(InputArray image, int r, OutputArray result);
void census_transform_binary(InputArray image, int r, OutputArray result);
uchar compare_value_binary(uchar center, uchar other);
int hamming_distance_bis(long left, long right);

#endif 
