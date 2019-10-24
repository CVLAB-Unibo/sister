#ifndef DSI_H
#define DSI_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

typedef struct _DSI
{
	int width;
	int height;
	int d_max;
	int d_min;
	int num_disp;
	bool similarity;
	vector<Mat> values;
};

_DSI normalize_DSI(_DSI DSI);
void write_DSI(_DSI DSI, char *filename);
_DSI read_DSI(char *filename);
_DSI read_DSI_binary(char *filename, int height, int width, int d_min, int d_max, bool similarity);
Mat disparity_map_L2R(_DSI DSI);
Mat disparity_map_R2L(_DSI DSI);
_DSI DSI_init(int height, int width, int d_min, int d_max, bool similarity);
_DSI DSI_left2right(_DSI DSI_L);
void write_disparity_map(InputArray disparity_map, char *file_name);
int readField(char buffer[], char sep, FILE *f);
void add_costs_array_DSI(_DSI DSI, int x, int y, float *costs_array);

#endif
