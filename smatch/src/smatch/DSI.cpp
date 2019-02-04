#include "DSI.h"

void write_DSI(_DSI DSI, char* filename)
{
	FILE * file = fopen(filename, "w+");

	if (file == NULL)
	{
		//fprintf(stderr, "Failed to open file [ %s ]\n", strerror(errno));
		exit(-1);
	}

	fprintf(file, "height %d width %d d_min %d d_max %d similarity %d\n", DSI.height, DSI.width, DSI.d_min, DSI.d_max, DSI.similarity);

	for (int row = 0; row < DSI.height; row++)
	{
		for (int col = 0; col < DSI.width; col++)
		{
			for (int d = 0; d < DSI.num_disp; d++)
			{
				fprintf(file, "%f ", DSI.values[d].ptr<float>(row)[col]);
			}

			fprintf(file, "\n");
		}
	}

	fclose(file);
}
_DSI read_DSI(char* filename)
{
	char BUFFER[3 + FLT_MANT_DIG - FLT_MIN_EXP];
	int width, height, d_min, d_max, similarity;
	int ok;

	FILE * file = fopen(filename, "r");

	if (file == NULL)
	{
		//fprintf(stderr, "Failed to open file [ %s ]\n", strerror(errno));
		exit(-1);
	}

	ok = fscanf(file, "height %d width %d d_min %d d_max %d  similarity %d\n", &height, &width, &d_min, &d_max, &similarity);

	_DSI DSI = DSI_init(height, width, d_min, d_max, similarity);

	for (int row = 0; row < DSI.height; row++)
	{
		for (int col = 0; col < DSI.width; col++)
		{
			for (int d = 0; d < DSI.num_disp -1; d++)
			{
				if (ok)
				{
					ok = readField(BUFFER, ' ', file);
					DSI.values[d].at<float>(row, col) = atof(BUFFER);
				}
			}

			if (ok)
			{
				ok = readField(BUFFER, '\n', file);
				DSI.values[d_max].at<float>(row, col) = atof(BUFFER);
			}
		}
	}

	fclose(file);

	return DSI;
}
_DSI read_DSI_binary(char* filename, int height, int width, int d_min, int d_max, bool similarity)
{
	_DSI DSI = DSI_init(height, width, d_min, d_max, similarity);

	FILE * file = fopen(filename, "rb");

	if (file == NULL)
	{
		//fprintf(stderr, "Failed to open file [ %s ]\n", strerror(errno));
		exit(-1);
	}

	for (int d = 0; d < DSI.num_disp; d++)
	{
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				float f;
				fread(&f, sizeof(float), 1, file);

				if (f != f)
				{
					DSI.values[d].ptr<float>(row)[col] = 2.0;
				}
				else
				{
					DSI.values[d].ptr<float>(row)[col] = 1.0 + f;
				}
			}
		}
	}

	fclose(file);

	return DSI;
}
_DSI normalize_DSI(_DSI DSI)
{
	_DSI DSI_out = DSI_init(DSI.height, DSI.width, DSI.d_min, DSI.d_max, DSI.similarity);

	float minimum = FLT_MAX, maximum = -FLT_MAX;

	for (int row = 0; row < DSI.height; row++)
	{
		for (int col = 0; col < DSI.width; col++)
		{
			for (int d = 0; d < DSI.num_disp - 1; d++)
			{
				float value = DSI.values[d].ptr<float>(row)[col];

				if (value > maximum) maximum = value;
				if (value < minimum) minimum = value;
			}
		}
	}

	for (int row = 0; row < DSI.height; row++)
	{
		for (int col = 0; col < DSI.width; col++)
		{
			for (int d = 0; d < DSI.num_disp - 1; d++)
			{
				DSI_out.values[d].ptr<float>(row)[col] = (DSI.values[d].ptr<float>(row)[col] - minimum) / (maximum - minimum);
			}
		}
	}

	return DSI_out;
}
Mat disparity_map_L2R(_DSI DSI)
{
	Mat disparity = Mat(DSI.height, DSI.width, CV_32F);

	float min_max;
	int WTA, disp_scale = (int)pow(2, 8) / DSI.num_disp;

	for (int row = 0; row < DSI.height; row++)
	{
		for (int col = 0; col < DSI.width; col++)
		{
			min_max = (DSI.similarity) ? 0 : FLT_MAX;
			WTA = 0;

			for (int d = 0; d < DSI.num_disp; d++)
			{
				float matching_cost = DSI.values[d].ptr<float>(row)[col]; 
		
				if (!DSI.similarity && min_max > matching_cost)
				{
					min_max = matching_cost; 
					WTA = d + DSI.d_min; 
				}
				else if (DSI.similarity && min_max < matching_cost)
				{
					min_max = matching_cost;
					WTA = d + DSI.d_min;
				}
			}

			if (WTA > 0) disparity.ptr<float>(row)[col] = (float)WTA * disp_scale; 
			else  disparity.ptr<float>(row)[col] = 0;
		}
	}

	return disparity;
}
Mat disparity_map_R2L(_DSI DSI)
{
	Mat disparity = Mat(DSI.height, DSI.width, CV_32F);

	float min_max;
	int WTA, disp_scale = (int)pow(2, 8) / DSI.num_disp;

	for (int row = 0; row < DSI.height; row++)
	{
		for (int col = 0; col < DSI.width; col++)
		{
			min_max = (DSI.similarity) ? 0 : FLT_MAX;
			WTA = 0;

			for (int d = 0; d < DSI.num_disp && col + d < DSI.width; d++)
			{
				float matching_cost = DSI.values[d].ptr<float>(row)[col + d];

				if (!DSI.similarity && min_max > matching_cost)
				{
					min_max = matching_cost;
					WTA = d + DSI.d_min;
				}
				else if (DSI.similarity && min_max < matching_cost)
				{
					min_max = matching_cost;
					WTA = d + DSI.d_min;
				}
			}

			if (WTA > 0) disparity.at<float>(row, col) = WTA * disp_scale;
			else  disparity.at<float>(row, col) = 0;
		}
	}

	return disparity;
}
_DSI DSI_init(int height, int width, int d_min, int d_max,  bool similarity)
{
	struct _DSI DSI;

	DSI.height = height;
	DSI.width = width;
	DSI.d_min = d_min;
	DSI.d_max = d_max;
	DSI.num_disp = d_max - d_min + 1;
	DSI.similarity = similarity;

	for (int d = 0; d < d_max - d_min + 1; d++)
	{
		DSI.values.push_back(Mat(height, width, CV_32F, Scalar(-1)));
	}

	return DSI;
}
_DSI DSI_left2right(_DSI DSI_L)
{
	int height = DSI_L.height;
	int width = DSI_L.width;
	int d_min = DSI_L.d_min;
	int d_max = DSI_L.d_max;
	int similarity = DSI_L.similarity;
	int num_disp = d_max - d_min + 1;

	_DSI DSI_R = DSI_init(height, width, d_min, d_max, similarity);

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			for (int d = 0; d < num_disp; d++)
			{
				if (col + d < width) //rivedi
					DSI_R.values[d].ptr<float>(row)[col] = DSI_L.values[d].ptr<float>(row)[col + d];
			}
		}
	}

	return DSI_R;
}
void write_disparity_map(InputArray disparity_map, char* file_name)
{
	Mat _temp = disparity_map.getMat();
	Mat _disparity_map; _disparity_map.create(_temp.size(), CV_8U);

	int height = _disparity_map.rows;
	int width = _disparity_map.cols;

	for (int row = 0; row < height; row++)
	{
		uchar* disparity_map_ptr = _disparity_map.ptr<uchar>(row);
		float* temp_ptr = _temp.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			if (temp_ptr[col] >= 0)
				disparity_map_ptr[col] = (uchar)temp_ptr[col];
			else
				disparity_map_ptr[col] = 0;
		}
	}

//	imwrite(file_name, _disparity_map);
}
int readField(char buffer[], char sep, FILE *f)
{
	int i = 0;
	char ch = fgetc(f);
	while (ch != sep && ch != 10 && ch != EOF) 
	{
		buffer[i] = ch;
		i++;
		ch = fgetc(f);
	}
	buffer[i] = '\0';
	return ch;
}
void add_costs_array_DSI(_DSI DSI, int x, int y, float* costs_array)
{
	for (int d = 0; d < DSI.num_disp; d++)
	{
		float value = DSI.values[d].ptr<float>(y)[x];

		if (value < 0)
			DSI.values[d].ptr<float>(y)[x] = costs_array[d];
		else
			DSI.values[d].ptr<float>(y)[x] += costs_array[d];
	}
}
